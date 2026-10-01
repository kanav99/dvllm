#pragma once

#include <cstdint>

#include "simdcrypt/SHA256.hpp"

typedef struct {
    int64_t dim; // transformer dimension
    int64_t hidden_dim; // for ffn layers
    int64_t n_layers; // number of layers
    int64_t n_heads; // number of query heads
    int64_t n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int64_t vocab_size; // vocabulary size, usually 256 (byte-level)
    int64_t seq_len; // max sequence length
} Config;


typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim * steps)
    float *xb; // same, but inside a residual branch (dim * steps)
    float *xb2; // an additional buffer just for convenience (dim * steps)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim * 2 * steps)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim * steps)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *qkv;
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p, int steps) {
    // we calloc instead of malloc to keep valgrind happy
    int64_t kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float *) calloc(p->dim * steps, sizeof(float));
    s->xb = (float *) calloc(p->dim * steps, sizeof(float));
    s->xb2 = (float *) calloc(p->dim * steps, sizeof(float));
    s->hb = (float *) calloc(p->hidden_dim * 2 * steps, sizeof(float));
    s->hb2 = s->hb + p->hidden_dim; 
    s->qkv = (float *) calloc((p->dim + 2 * kv_dim) * steps, sizeof(float));
    s->q = s->qkv; // q is the first part of qkv
    s->key_cache =(float *) calloc(p->n_layers * steps * kv_dim, sizeof(float));
    s->value_cache = (float *) calloc(p->n_layers * steps * kv_dim, sizeof(float));
    s->att = (float *) calloc(p->n_heads * steps, sizeof(float));
    s->logits = (float *) calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->qkv);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->rms_final_weight = ptr;
}

void read_checkpoint(const char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, const char* checkpoint_path, int &steps) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    if (steps == 0 || steps > t->config.seq_len) steps = t->config.seq_len; // override to ~max length
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config, steps);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

#include "constants.hpp"
#include "cache.hpp"
#include "async_verify.hpp"
#include "networking.hpp"

// Instantiate the global cache
LRUEmbeddingCache embed_cache;
uint64_t total_embedding_calls = 0;
uint64_t total_embedding_hits = 0;

// stats
uint64_t rounds = 0;

char *tmp_buffer = nullptr;
integer_t *qx = nullptr;
integer_t *qy = nullptr;
long matmul_time = 0;
long prng_time = 0;
long eat_time = 0;
int num_servers = 0;
simdcrypt::PRNG *prng = nullptr;
Verifier *verifier = nullptr;
AsyncVerifier *async_verifier = nullptr;

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void matmul_remote(std::vector<int> &sockfds, float* xout, float* x, int n, int d, int layer, int ID, int batch_size = 1) {
    // W (d,n) @ x (n,) -> xout (d,)
    long start = time_in_ms();

    int stride = std::max(n * batch_size, d * batch_size) * (int)sizeof(integer_t);
    int send_size = n * batch_size * sizeof(integer_t);
    // The server now accumulates the per-chunk partial sums on its own CPU, so we
    // receive a single value per output element instead of one per chunk.
    int recv_size = d * batch_size * sizeof(integer_t);
    
    integer_t *payload0 = reinterpret_cast<integer_t *>(tmp_buffer);
    for (int i = 0; i < n * batch_size; i++) {
        payload0[i] = (integer_t)(x[i] * FP_SCALE);
        qx[i] = payload0[i];
    }
    
    long prng_start = time_in_ms();
    if (num_servers > 1) {
        for (int party = 1; party < num_servers; party++) {
            integer_t *payload = reinterpret_cast<integer_t *>(tmp_buffer + party * stride);

            prng[party].get(payload, n * batch_size);
        }
    }
    prng_time += time_in_ms() - prng_start;
    
    for (int party = 1; party < num_servers; party++) {
        integer_t *payload = reinterpret_cast<integer_t *>(tmp_buffer + party * stride);
        for (int i = 0; i < n * batch_size; i++) {
            payload0[i] = payload0[i] - payload[i];
        }
    }

    std::vector<std::thread> workers;
    for (int idx = 0; idx < num_servers; idx++) {
        workers.emplace_back([&sockfds, idx, send_size, recv_size, stride]() {
            if (idx == 0) {
                write_into(sockfds[idx], tmp_buffer + idx * stride, send_size);
            }
            read_into(sockfds[idx], (char *)(tmp_buffer + idx * stride), recv_size);
        });
    }

    // Join all threads to synchronize before processing the results
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    rounds += 1;
    
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < d; i++) {
            int64_t out = 0; // Collect the additive server shares in 64-bit to avoid summation overflow
            for (int j = 0; j < num_servers; j++) {
                integer_t* recv_buf = (integer_t*)(tmp_buffer + j * stride);
                out += recv_buf[b * d + i];
            }
            qy[b * d + i] = (integer_t)out; // Re-align to expected int32 layout for the verifier routine
            xout[b * d + i] = float(out) / (FP_SCALE * FP_W_SCALE);
        }
    }
    long eat_start = time_in_ms();
    for (int b = 0; b < batch_size; b++) {
        async_verifier->push(layer, ID, qx + b * n, n, qy + b * d, d);
    }
    eat_time += time_in_ms() - eat_start;
    matmul_time += time_in_ms() - start;
}

char *embed_in_cache = nullptr;
size_t embed_in_cache_size = 0;
int32_t *embed_out_cache = nullptr;
simdcrypt::SHA256 hasher;
uint8_t* embedding_commitment = nullptr;
int64_t embed_time = 0;

void one_hot_encode(int token, int vocab_size, char *output) {
    memset(output, 0, (vocab_size + 7) / 8);
    output[token / 8] |= 1 << (token % 8);
}

void embed_remote_batch(std::vector<int> &sockfds, int* token, float* embedding, int dim, int vocab_size, int batch_size = 1) {
    int64_t start = time_in_ms();
    total_embedding_calls += batch_size;

    char *embed_for_server0 = embed_in_cache;
    for (int i = 0; i < batch_size; i++) {
        one_hot_encode(token[i], vocab_size, embed_for_server0 + i * ((vocab_size + 7) / 8));
    }
    
    for (int i = 1; i < num_servers; i++) {
        char *embed_for_server = embed_in_cache + i * embed_in_cache_size;
        prng[i].get(embed_for_server, batch_size * ((vocab_size + 7) / 8));

        #pragma omp simd
        for (int j = 0; j < batch_size * ((vocab_size + 7) / 8); j++) {
            embed_for_server0[j] ^= embed_for_server[j];
        }
    }

    std::vector<std::thread> workers;
    for (int i = 0; i < num_servers; i++) {
        workers.emplace_back([=]() {
            if (i == 0) write_into(sockfds[i], embed_in_cache + i * embed_in_cache_size, batch_size * ((vocab_size + 7) / 8));
            read_into(sockfds[i], (char *)(embed_out_cache + i * batch_size * dim), batch_size * dim * sizeof(int32_t));
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    rounds += 1;

    memset(embedding, 0, batch_size * dim * sizeof(float));
    for (int i = 0; i < batch_size * dim; i++) {
        uint32_t embed_int = 0;
        for (int j = 0; j < num_servers; j++) {
            embed_int ^= embed_out_cache[j * batch_size * dim + i];
        }
        embedding[i] = *(float *)(&embed_int);
    }
    
    uint8_t hash[simdcrypt::SHA256::HashSize];
    for (int i = 0; i < batch_size; i++) {
        hasher.Hash((uint8_t *)(embedding + i * dim), dim * sizeof(float), hash);
        if (memcmp(hash, embedding_commitment + token[i] * simdcrypt::SHA256::HashSize, simdcrypt::SHA256::HashSize) != 0) {
            std::cerr << "Error: Embedding commitment does not match for token " << token[i] << std::endl;
        }
        embed_cache.put(token[i], embedding + i * dim, dim);
    }
    embed_time += time_in_ms() - start;
}

void embed_remote(std::vector<int> &sockfds, int token, float* embedding, int dim, int vocab_size) {
    int64_t start = time_in_ms();
    total_embedding_calls++;
    // 1. Check the cache first
    if (embed_cache.get(token, embedding, dim)) {
        total_embedding_hits++;
        return; // Cache hit! Skip all remote calls.
    }

    // --- Cache Miss: Proceed with remote retrieval ---
    char *embed_for_server0 = embed_in_cache;
    one_hot_encode(token, vocab_size, embed_for_server0);
    
    for (int i = 1; i < num_servers; i++) {
        char *embed_for_server = embed_in_cache + i * embed_in_cache_size;// + 2 * sizeof(int);
        prng[i].get(embed_for_server, (vocab_size + 7) / 8);
    }

    for (int i = 1; i < num_servers; i++) {
        char *embed_for_server = embed_in_cache + i * embed_in_cache_size;// + 2 * sizeof(int);
        for (int j = 0; j < (vocab_size + 7) / 8; j++) {
            embed_for_server0[j] ^= embed_for_server[j];
        }
    }

    // #pragma omp parallel for
    std::vector<std::thread> workers;
    for (int i = 0; i < num_servers; i++) {
        workers.emplace_back([=]() {
            if (i == 0) write_into(sockfds[i], embed_in_cache + i * embed_in_cache_size, (vocab_size + 7) / 8);
            read_into(sockfds[i], (char *)(embed_out_cache + i * dim), dim * sizeof(int32_t));
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    rounds += 1;

    memset(embedding, 0, dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        uint32_t embed_int = 0;
        for (int j = 0; j < num_servers; j++) {
            embed_int ^= embed_out_cache[j * dim + i];
        }
        embedding[i] = *(float *)(&embed_int);
    }

    uint8_t hash[simdcrypt::SHA256::HashSize];
    hasher.Hash((uint8_t *)embedding, dim * sizeof(float), hash);
    if (memcmp(hash, embedding_commitment + token * simdcrypt::SHA256::HashSize, simdcrypt::SHA256::HashSize) != 0) {
        std::cerr << "Error: Embedding commitment does not match for token " << token << std::endl;
    }

    // 2. Add the verified embedding to the cache so future accesses are instantaneous
    embed_cache.put(token, embedding, dim);
    embed_time += time_in_ms() - start;
}

float* forward(Transformer* transformer, int token, int pos, int steps, std::vector<int> &sockfds) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int64_t dim = p->dim;
    int64_t kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int64_t kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int64_t hidden_dim =  p->hidden_dim;
    int64_t head_size = dim / p->n_heads;

    embed_remote(sockfds, token, x, dim, p->vocab_size);

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int64_t loff = l * steps * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        if (((p->n_heads + 2 * p->n_kv_heads) * head_size) != (dim + 2 * kv_dim)) {
            fprintf(stderr, "Error: Dimension mismatch in matmul for qkv\n");
            exit(EXIT_FAILURE);
        }

        // qkv matmuls for this position
        matmul_remote(sockfds, s->qkv, s->xb, dim, dim + 2 * kv_dim, l, 8);
        // split qkv into q, k, v
        memcpy(s->k, s->qkv + dim, kv_dim * sizeof(float));
        memcpy(s->v, s->qkv + dim + kv_dim, kv_dim * sizeof(float));

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(ROPE_CONST, head_dim / (float)head_size); // Llama 2 Base Frequency
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;
        // #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * steps;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul_remote(sockfds, s->xb2, s->xb, dim, dim, l, 3);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        // matmul_remote(sockfds, s->hb, s->xb, dim, hidden_dim, l, 4);
        // matmul_remote(sockfds, s->hb2, s->xb, dim, hidden_dim, l, 6);
        matmul_remote(sockfds, s->hb, s->xb, dim, 2 * hidden_dim, l, 9);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul_remote(sockfds, s->xb, s->hb, hidden_dim, dim, l, 5);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul_remote(sockfds, s->logits, x, p->dim, p->vocab_size, 0, 7);
    return s->logits;
}

// Batched forward pass (Prefill)
float* forward_batch(Transformer* transformer, int* tokens, int pos_start, int batch_size, int steps, std::vector<int> &sockfds) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int64_t dim = p->dim;
    int64_t kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int64_t kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int64_t hidden_dim =  p->hidden_dim;
    int64_t head_size = dim / p->n_heads;

    embed_remote_batch(sockfds, tokens, x, dim, p->vocab_size, batch_size);

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        for (int b = 0; b < batch_size; b++) {
            rmsnorm(s->xb + b * dim, x + b * dim, w->rms_att_weight + l*dim, dim);
        }

        // qkv matmuls for this position
        matmul_remote(sockfds, s->qkv, s->xb, dim, dim + 2 * kv_dim, l, 8, batch_size);

        // key and value point to the kv cache
        int64_t loff = l * steps * kv_dim; // kv cache layer offset for convenience
        for (int b = 0; b < batch_size; b++) {
            int pos = pos_start + b;
            float* qkv_b = s->qkv + b * (dim + 2 * kv_dim);
            float* q_b = qkv_b;
            float* k_b = s->key_cache + loff + pos * kv_dim;
            float* v_b = s->value_cache + loff + pos * kv_dim;

            // split qkv into q, k, v
            memcpy(k_b, qkv_b + dim, kv_dim * sizeof(float));
            memcpy(v_b, qkv_b + dim + kv_dim, kv_dim * sizeof(float));

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i+=2) {
                int head_dim = i % head_size;
                float freq = 1.0f / powf(ROPE_CONST, head_dim / (float)head_size); 
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    float* vec = v == 0 ? q_b : k_b; // the vector to rotate (query or key)
                    float v0 = vec[i];
                    float v1 = vec[i+1];
                    vec[i]   = v0 * fcr - v1 * fci;
                    vec[i+1] = v0 * fci + v1 * fcr;
                }
            }
        }

        // multihead attention. iterate over all heads
        for (int b = 0; b < batch_size; b++) {
            int pos = pos_start + b;
            float* q_b = s->qkv + b * (dim + 2 * kv_dim);
            for (int h = 0; h < p->n_heads; h++) {
                // get the query vector for this head
                float* q_head = q_b + h * head_size;
                // attention scores for this head
                float* att = s->att + h * steps;

                // iterate over all timesteps, including the current one
                for (int tau = 0; tau <= pos; tau++) {
                    // get the key vector for this head and at this timestep
                    float* k_head = s->key_cache + loff + tau * kv_dim + (h / kv_mul) * head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += q_head[i] * k_head[i];
                    }
                    score /= sqrtf(head_size);
                    // save the score to the attention buffer
                    att[tau] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(att, pos + 1);

                // weighted sum of the values, store back into xb
                float* xb_head = s->xb + b * dim + h * head_size;
                memset(xb_head, 0, head_size * sizeof(float));
                for (int tau = 0; tau <= pos; tau++) {
                    // get the value vector for this head and at this timestep
                    float* v_head = s->value_cache + loff + tau * kv_dim + (h / kv_mul) * head_size;
                    // get the attention weight for this timestep
                    float a = att[tau];
                    // accumulate the weighted value into xb
                    for (int i = 0; i < head_size; i++) {
                        xb_head[i] += a * v_head[i];
                    }
                }
            }
        }

        // final matmul to get the output of the attention
        matmul_remote(sockfds, s->xb2, s->xb, dim, dim, l, 3, batch_size);

        // residual connection back into x
        for (int i = 0; i < batch_size * dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        for (int b = 0; b < batch_size; b++) {
            rmsnorm(s->xb + b * dim, x + b * dim, w->rms_ffn_weight + l*dim, dim);
        }

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul_remote(sockfds, s->hb, s->xb, dim, 2 * hidden_dim, l, 9, batch_size);

        // SwiGLU non-linearity
        // [FIX] Compact SwiGLU results to s->hb + b * hidden_dim so Matmul ID 5 reads a contiguous array
        for (int b = 0; b < batch_size; b++) {
            float* hb_src = s->hb + b * (2 * hidden_dim);
            float* hb2_src = hb_src + hidden_dim;
            float* hb_dst = s->hb + b * hidden_dim;
            for (int i = 0; i < hidden_dim; i++) {
                float val = hb_src[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= (1.0f / (1.0f + expf(-val)));
                // elementwise multiply with w3(x)
                val *= hb2_src[i];
                hb_dst[i] = val;
            }
        }

        // final matmul to get the output of the ffn
        matmul_remote(sockfds, s->xb, s->hb, hidden_dim, dim, l, 5, batch_size);

        // residual connection
        for (int i = 0; i < batch_size * dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    float* last_x = x + (batch_size - 1) * dim;
    rmsnorm(last_x, last_x, w->rms_final_weight, dim);

    // classifier into logits
    matmul_remote(sockfds, s->logits, last_x, p->dim, p->vocab_size, 0, 7, 1);
    
    return s->logits;
}
