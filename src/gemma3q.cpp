/* Inference for Gemma 3 Transformer model in pure C++ (delegated, quantized) */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <algorithm>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string>
#include "simdcrypt/PRNG.hpp"
#include "simdcrypt/SHA256.hpp"
#include <fstream>

#include "client_gemma3.hpp"

const int GEMMA_BOS = 2;
const int GEMMA_EOS = 1;
const int GEMMA_END_OF_TURN = 106;

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
//
// Gemma 3's tokenizer.json is a HF fast-tokenizer BPE model (vocab + merges,
// rank = list index), exported by models/gemma3_tok.py into the same
// tokenizer.bin binary format llama2_tok.py/llama3.2_tok.py already use.
// Unlike Llama 2, Gemma doesn't add a dummy leading-space prefix, and
// unlike Llama 3.2 it doesn't need a full regex pre-tokenizer -- the
// normalizer already turns spaces into the sentencepiece '▁' marker (baked
// into the exported vocab as literal spaces), so a single whole-string
// byte-scan + greedy merge reproduces the reference tokenizer's output
// (verified empirically against `tokenizers.Tokenizer.from_file(...)`).
// Merges are ranked ascending (lower = merge first), same convention as
// llama3.2q.cpp. The one thing that can't be reached by merging alone is
// special tokens like <start_of_turn>/<end_of_turn>, so those are
// intercepted with a small bracket scanner before BPE runs on the rest.

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;

    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int token) {
    char *piece = t->vocab[token];
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        static unsigned char byte_piece[2];
        byte_piece[0] = byte_val;
        byte_piece[1] = '\0';
        piece = (char*)byte_piece;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = (TokenIndex *) bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

// byte-scan + greedy BPE merge (ascending rank) over one plain-text segment;
// appends resulting token ids directly onto the shared tokens[] array
void bpe_encode_segment(Tokenizer* t, const char* seg, size_t seg_len, char* str_buffer, int* tokens, int* n_tokens) {
    if (seg_len == 0) return;
    int seg_tok_start = *n_tokens;
    size_t str_len = 0;

    for (size_t ci = 0; ci < seg_len; ci++) {
        unsigned char c = (unsigned char)seg[ci];
        if ((c & 0xC0) != 0x80) { str_len = 0; }
        str_buffer[str_len++] = (char)c;
        str_buffer[str_len] = '\0';

        bool next_is_continuation = (ci + 1 < seg_len) && (((unsigned char)seg[ci+1] & 0xC0) == 0x80);
        if (next_is_continuation && str_len < 4) { continue; }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback: Gemma's byte tokens are named "<0xXX>" vocab
            // entries (not at a fixed offset like Llama 2's `byte + 3`)
            for (size_t bi = 0; bi < str_len; bi++) {
                char byte_tok[8];
                snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)str_buffer[bi]);
                int bid = str_lookup(byte_tok, t->sorted_vocab, t->vocab_size);
                if (bid == -1) {
                    fprintf(stderr, "missing byte-fallback token for byte 0x%02x\n", (unsigned char)str_buffer[bi]);
                    exit(EXIT_FAILURE);
                }
                tokens[(*n_tokens)++] = bid;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = 1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = seg_tok_start; i < (*n_tokens) - 1; i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] < best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) { break; }

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens) - 1; i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex *) malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = (char *)malloc((t->max_token_length*2 + 1 + 2) * sizeof(char));

    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = GEMMA_BOS;

    size_t text_len = strlen(text);
    size_t idx = 0;
    size_t seg_start = 0;

    while (idx < text_len) {
        if (text[idx] == '<') {
            size_t close = idx + 1;
            while (close < text_len && text[close] != '<' && text[close] != '>'
                   && (close - idx) < t->max_token_length) {
                close++;
            }
            if (close < text_len && text[close] == '>') {
                size_t cand_len = close - idx + 1;
                memcpy(str_buffer, text + idx, cand_len);
                str_buffer[cand_len] = '\0';
                int special_id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
                if (special_id != -1) {
                    bpe_encode_segment(t, text + seg_start, idx - seg_start, str_buffer, tokens, n_tokens);
                    tokens[(*n_tokens)++] = special_id;
                    idx = close + 1;
                    seg_start = idx;
                    continue;
                }
            }
        }
        idx++;
    }
    bpe_encode_segment(t, text + seg_start, text_len - seg_start, str_buffer, tokens, n_tokens);

    if (eos) tokens[(*n_tokens)++] = GEMMA_EOS;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = (ProbIndex *) malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, int* prompt_tokens, int num_prompt_tokens, int steps, std::vector<int> &sockfds) {

    long start = time_in_ms();
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    int T = std::min(num_prompt_tokens, steps);

    if (T > 0) {
        float* logits = forward_batch(transformer, prompt_tokens, 0, T, steps, sockfds);
        next = sample(sampler, logits);

        for (int i = 1; i < T; i++) {
            char* piece = decode(tokenizer, prompt_tokens[i]);
            safe_printf(piece);
        }

        char* piece = decode(tokenizer, next);
        safe_printf(piece);
        fflush(stdout);

        token = next;
        pos = T;
    }

    while (pos < steps) {
        float* logits = forward(transformer, token, pos, steps, sockfds);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == GEMMA_EOS || next == GEMMA_END_OF_TURN) { break; }

        char* piece = decode(tokenizer, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", pos / (double)(end-start)*1000);
        fprintf(stderr, "total time: %f\n", (end-start)/1000.0);
        fprintf(stderr, "total matmul time: %f\n", matmul_time/1000.0);
        fprintf(stderr, "total prng time: %f\n", prng_time/1000.0);
        fprintf(stderr, "total eat time: %f\n", eat_time/1000.0);
        fprintf(stderr, "total embed time: %f\n", embed_time/1000.0);
        fprintf(stderr, "total embedding calls: %llu\n", total_embedding_calls);
        fprintf(stderr, "total embedding hits: %llu\n", total_embedding_hits);
    }
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
    }
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   gemma3q <model_name> [options]\n");
    fprintf(stderr, "Example: gemma3q gemma-3-1b-it -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -r <string> address of server as <ip> or <ip>:<port> (default port %d), can be set multiple times for secure protocol\n", DEFAULT_PORT);
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    char *model_name = NULL;
    char *tokenizer_path = "gemma3.tokenizer";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    std::vector<std::string> ip_addresses;
    std::vector<int> ip_ports;

    if (argc < 2) {
        error_usage();
    }

    if (argc >= 2) { model_name = argv[1]; }
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'r') {
            std::string ip; int port;
            if (!parse_endpoint(argv[i + 1], ip, port)) { error_usage(); }
            ip_addresses.push_back(ip); ip_ports.push_back(port);
        }
        else { error_usage(); }
    }

    num_servers = ip_addresses.size();
    std::cerr << "Using " << num_servers << " servers" << std::endl;

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;
    std::cout << "Using seed: " << rng_seed << std::endl;

    Transformer transformer;
    std::string checkpoint_path = std::string(getenv("HOME")) + std::string("/.dvllm/") + model_name + std::string("/client.bin");
    build_transformer(&transformer, checkpoint_path.c_str(), steps);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    char *empty_prompt = (char *)"";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        num_prompt_tokens = 1;
    }

    int64_t q_dim = transformer.config.n_heads * transformer.config.head_dim;
    int64_t kv_dim = transformer.config.n_kv_heads * transformer.config.head_dim;
    size_t max_matmul_elements = std::max({
        (int64_t)transformer.config.dim,
        (int64_t)transformer.config.hidden_dim,
        (int64_t)transformer.config.vocab_size,
        q_dim + 2 * kv_dim,
        2 * (int64_t)transformer.config.hidden_dim,
        q_dim
    });
    int max_n = std::max({(int64_t)transformer.config.dim, (int64_t)transformer.config.hidden_dim, q_dim});
    int max_chunks = (max_n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int max_batch = std::max(1, num_prompt_tokens);
    size_t max_buffer_elements = std::max(max_matmul_elements, max_matmul_elements * max_chunks) * max_batch;

    tmp_buffer = new char[num_servers * (max_buffer_elements * sizeof(integer_t) + 2 * sizeof(int))];
    qx = new integer_t[max_matmul_elements * max_batch];
    qy = new integer_t[max_matmul_elements * max_batch];

    simdcrypt::block seeds[num_servers];
    prng = new simdcrypt::PRNG[num_servers];
    for (int i = 0; i < num_servers; i++) {
        seeds[i] = simdcrypt::toBlock(rand(), rand());
        prng[i] = simdcrypt::PRNG(seeds[i]);
    }
    verifier = new Verifier(std::string(getenv("HOME")) + std::string("/.dvllm/") + model_name + std::string("/commits"));
    async_verifier = new AsyncVerifier(verifier);
    embed_in_cache_size = max_batch * ((transformer.config.vocab_size + 7) / 8);
    embed_in_cache = new char[num_servers * embed_in_cache_size];
    memset(embed_in_cache, 0, num_servers * embed_in_cache_size);
    embed_out_cache = new int32_t[max_batch * num_servers * transformer.config.dim];

    std::string embed_path = std::string(getenv("HOME")) + std::string("/.dvllm/") + model_name + std::string("/commits/embeddings.bin");
    embedding_commitment = new uint8_t[transformer.config.vocab_size * simdcrypt::SHA256::HashSize];
    std::ifstream embed_file(embed_path, std::ios::binary);
    embed_file.read(reinterpret_cast<char*>(embedding_commitment), transformer.config.vocab_size * simdcrypt::SHA256::HashSize);
    embed_file.close();

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    std::vector<int> sockfds;
    for (const auto& ip : ip_addresses) {
        sockfds.push_back(connect(ip, ip_ports[&ip - &ip_addresses[0]]));
    }

    for (int i = 0; i < sockfds.size(); i++) {
        data_sent[sockfds[i]] = 0;
        data_received[sockfds[i]] = 0;
        comm_time_in_ms[sockfds[i]] = 0;
    }

    int server_mode = 0;
    write_into(sockfds[0], server_mode);
    write_into(sockfds[0], steps);
    write_into(sockfds[0], num_prompt_tokens);
    server_mode = 1;
    for (int i = 1; i < sockfds.size(); i++) {
        write_into(sockfds[i], server_mode);
        write_into(sockfds[i], seeds[i]);
        write_into(sockfds[i], steps);
        write_into(sockfds[i], num_prompt_tokens);
    }

    generate(&transformer, &tokenizer, &sampler, prompt_tokens, num_prompt_tokens, steps, sockfds);

    int stop_layer = -1;
    write_into(sockfds[0], stop_layer);
    for (int sockfd : sockfds) {
        close(sockfd);
    }

    async_verifier->finish_and_stop();
    printf("Loading commits from disk...");
    fflush(stdout);
    verifier->Load();  // reads setup/commit files; excluded from the timed verification below
    printf("done.\n");
    printf("Verifying this session...\n");
    auto verification_start = time_in_ms();
    verifier->Verify();
    auto verification_end = time_in_ms();
    printf("Verification completed in %f s.\n", (verification_end - verification_start) / 1000.0);

    printf("Communication times:\n");
    for (const auto& pair : comm_time_in_ms) {
        printf("  Socket %d: %llu ms\n", pair.first, pair.second);
    }
    printf("Data sent:\n");
    for (const auto& pair : data_sent) {
        printf("  Socket %d: %llu bytes\n", pair.first, pair.second);
    }
    printf("Data received:\n");
    for (const auto& pair : data_received) {
        printf("  Socket %d: %llu bytes\n", pair.first, pair.second);
    }
    printf("Rounds completed: %llu\n", rounds);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    free(prompt_tokens);
    delete[] qx;
    delete[] qy;
    delete[] embed_in_cache;
    delete[] embed_out_cache;
    delete[] prng;
    delete async_verifier;
    delete verifier;
    return 0;
}
#endif
