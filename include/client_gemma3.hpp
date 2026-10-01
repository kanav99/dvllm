#pragma once

#include <cstdint>

#include "simdcrypt/SHA256.hpp"

// Gemma 3 analogue of client.hpp. See that file for the Llama version this
// is adapted from, and include/common_gemma3.hpp for the on-disk format
// shared with the quantizer/server. Kept as its own self-contained header
// (not #including common_gemma3.hpp) to match the existing convention of
// client.hpp/common.hpp each carrying their own (structurally identical)
// copy of Config.

typedef struct {
    int64_t dim;
    int64_t hidden_dim;
    int64_t n_layers;
    int64_t n_heads;
    int64_t n_kv_heads;
    int64_t vocab_size;
    int64_t seq_len;
    int64_t head_dim;
    int64_t sliding_window;
    int64_t sliding_window_pattern;
    double query_pre_attn_scalar;
    double rope_theta_local;
    double rope_theta_global;
    double rope_scaling_factor; // linear RoPE scaling applied to global layers only (1.0 = none)
} Config;

inline bool is_global_layer(const Config* p, int64_t layer) {
    return ((layer + 1) % p->sliding_window_pattern) == 0;
}

typedef struct {
    // token embedding table pointer is unused client-side (embeddings are
    // fetched obliviously over the network, see embed_remote below) --
    // kept only for structural parity with common_gemma3.hpp.
    float* token_embedding_table;
    // sandwich-norm weights (layer, dim), already baked with +1.0 at export time
    float* rms_att_weight;
    float* rms_post_att_weight;
    float* rms_pre_ffn_weight;
    float* rms_post_ffn_weight;
    // QK-norm weights (layer, head_dim)
    float* rms_q_weight;
    float* rms_k_weight;
    // final rmsnorm
    float* rms_final_weight;
} TransformerWeights;

typedef struct {
    float *x;   // activation at current time stamp (dim * steps)
    float *xb;  // norm scratch (dim * steps)
    float *xa;  // attention-head output, pre-wo (q_dim * steps)
    float *xb2; // wo/mlp output, added into residual (dim * steps)
    float *hb;  // ffn hidden buffer (hidden_dim * 2 * steps)
    float *hb2;
    float *q;   // query (q_dim,)
    float *k;
    float *v;
    float *qkv;
    float *att;    // attention scores (n_heads, steps)
    float *logits; // output logits
    float* key_cache;   // (layer, steps, kv_dim)
    float* value_cache; // (layer, steps, kv_dim)
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

void malloc_run_state(RunState* s, Config* p, int steps) {
    int64_t q_dim = p->n_heads * p->head_dim;
    int64_t kv_dim = p->n_kv_heads * p->head_dim;
    int64_t max_dim = std::max(p->dim, q_dim);
    s->x = (float *) calloc(p->dim * steps, sizeof(float));
    s->xb = (float *) calloc(max_dim * steps, sizeof(float));
    s->xa = (float *) calloc(q_dim * steps, sizeof(float));
    s->xb2 = (float *) calloc(p->dim * steps, sizeof(float));
    s->hb = (float *) calloc(p->hidden_dim * 2 * steps, sizeof(float));
    s->hb2 = s->hb + p->hidden_dim;
    s->qkv = (float *) calloc((q_dim + 2 * kv_dim) * steps, sizeof(float));
    s->q = s->qkv; // q is the first part of qkv
    s->key_cache = (float *) calloc(p->n_layers * steps * kv_dim, sizeof(float));
    s->value_cache = (float *) calloc(p->n_layers * steps * kv_dim, sizeof(float));
    s->att = (float *) calloc(p->n_heads * steps, sizeof(float));
    s->logits = (float *) calloc(p->vocab_size, sizeof(float));
    if (!s->x || !s->xb || !s->xa || !s->xb2 || !s->hb || !s->qkv
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xa);
    free(s->xb2);
    free(s->hb);
    free(s->qkv);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    unsigned long long n_layers = p->n_layers;
    w->rms_att_weight = ptr;      ptr += n_layers * p->dim;
    w->rms_post_att_weight = ptr; ptr += n_layers * p->dim;
    w->rms_pre_ffn_weight = ptr;  ptr += n_layers * p->dim;
    w->rms_post_ffn_weight = ptr; ptr += n_layers * p->dim;
    w->rms_q_weight = ptr;        ptr += n_layers * p->head_dim;
    w->rms_k_weight = ptr;        ptr += n_layers * p->head_dim;
    w->rms_final_weight = ptr;
}

void read_checkpoint(const char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    int shared_weights;
    if (fread(&shared_weights, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);
    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + (sizeof(Config) + sizeof(int)) / sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, const char* checkpoint_path, int &steps) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    if (steps == 0 || steps > t->config.seq_len) steps = t->config.seq_len;
    malloc_run_state(&t->state, &t->config, steps);
}

void free_transformer(Transformer* t) {
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // Gemma's RMSNorm scales by (1 + weight); the +1.0 is baked into
    // `weight` at export time (models/gemma3.py), so this is otherwise
    // identical to the Llama version in client.hpp.
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-6f; // Gemma 3 uses rms_norm_eps=1e-6, vs Llama's 1e-5
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

float gelu_tanh(float x) {
    // hidden_activation="gelu_pytorch_tanh"
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

#include "constants.hpp"
#include "cache.hpp"
#include "async_verify.hpp"
#include "networking.hpp"

// Instantiate the global cache
LRUEmbeddingCache embed_cache;
uint64_t total_embedding_calls = 0;
uint64_t total_embedding_hits = 0;

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
            qy[b * d + i] = (integer_t)out;
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
    if (embed_cache.get(token, embedding, dim)) {
        total_embedding_hits++;
        return;
    }

    char *embed_for_server0 = embed_in_cache;
    one_hot_encode(token, vocab_size, embed_for_server0);

    for (int i = 1; i < num_servers; i++) {
        char *embed_for_server = embed_in_cache + i * embed_in_cache_size;
        prng[i].get(embed_for_server, (vocab_size + 7) / 8);
    }

    for (int i = 1; i < num_servers; i++) {
        char *embed_for_server = embed_in_cache + i * embed_in_cache_size;
        for (int j = 0; j < (vocab_size + 7) / 8; j++) {
            embed_for_server0[j] ^= embed_for_server[j];
        }
    }

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

    embed_cache.put(token, embedding, dim);
    embed_time += time_in_ms() - start;
}

// applies QK-norm then RoPE to a single head's q or k vector, in place.
// `scale` is the linear RoPE scaling factor (divides the rotation angle);
// 1.0 for models/layers with no scaling, e.g. Gemma 3 1B/270M, or any local
// (sliding-window) layer -- only global layers on the 4B/12B/27B sizes use it.
void norm_and_rotate_head(float* vec, float* norm_weight, int head_dim, int pos, double theta, double scale) {
    rmsnorm(vec, vec, norm_weight, head_dim);
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf((float)theta, (float)i / (float)head_dim);
        float val = (pos * freq) / (float)scale;
        float fcr = cosf(val);
        float fci = sinf(val);
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i]   = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
    }
}

float* forward(Transformer* transformer, int token, int pos, int steps, std::vector<int> &sockfds) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int64_t dim = p->dim;
    int64_t head_dim = p->head_dim;
    int64_t q_dim = p->n_heads * head_dim;
    int64_t kv_dim = p->n_kv_heads * head_dim;
    int64_t kv_mul = p->n_heads / p->n_kv_heads;
    int64_t hidden_dim = p->hidden_dim;
    float attn_scale = 1.0f / sqrtf((float)p->query_pre_attn_scalar);

    embed_remote(sockfds, token, x, dim, p->vocab_size);
    float embed_scale = sqrtf((float)dim);
    for (int i = 0; i < dim; i++) x[i] *= embed_scale;

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        bool global_layer = is_global_layer(p, l);
        double theta = global_layer ? p->rope_theta_global : p->rope_theta_local;
        double rope_scale = global_layer ? p->rope_scaling_factor : 1.0;

        // attention rmsnorm (input_layernorm)
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        int64_t loff = l * steps * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul_remote(sockfds, s->qkv, s->xb, dim, q_dim + 2 * kv_dim, l, 8);
        memcpy(s->k, s->qkv + q_dim, kv_dim * sizeof(float));
        memcpy(s->v, s->qkv + q_dim + kv_dim, kv_dim * sizeof(float));

        // QK-norm then RoPE, per head
        for (int h = 0; h < p->n_heads; h++) {
            norm_and_rotate_head(s->q + h*head_dim, w->rms_q_weight + l*head_dim, head_dim, pos, theta, rope_scale);
        }
        for (int h = 0; h < p->n_kv_heads; h++) {
            norm_and_rotate_head(s->k + h*head_dim, w->rms_k_weight + l*head_dim, head_dim, pos, theta, rope_scale);
        }

        int64_t window_start = global_layer ? 0 : std::max((int64_t)0, (int64_t)pos - p->sliding_window + 1);

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_dim;
            float* att = s->att + h * steps;
            for (int t = window_start; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q[i] * k[i];
                }
                score *= attn_scale;
                att[t] = score;
            }

            softmax(att + window_start, pos - window_start + 1);

            float* xa = s->xa + h * head_dim;
            memset(xa, 0, head_dim * sizeof(float));
            for (int t = window_start; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                float a = att[t];
                for (int i = 0; i < head_dim; i++) {
                    xa[i] += a * v[i];
                }
            }
        }

        // wo: attention output (q_dim) -> dim
        matmul_remote(sockfds, s->xb2, s->xa, q_dim, dim, l, 3);
        // post-attention norm, then residual add
        rmsnorm(s->xb2, s->xb2, w->rms_post_att_weight + l*dim, dim);
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // pre-feedforward norm
        rmsnorm(s->xb, x, w->rms_pre_ffn_weight + l*dim, dim);

        matmul_remote(sockfds, s->hb, s->xb, dim, 2 * hidden_dim, l, 9);

        // GeGLU: gelu(gate) * up
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = gelu_tanh(s->hb[i]) * s->hb2[i];
        }

        matmul_remote(sockfds, s->xb2, s->hb, hidden_dim, dim, l, 5);
        // post-feedforward norm, then residual add
        rmsnorm(s->xb2, s->xb2, w->rms_post_ffn_weight + l*dim, dim);
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
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
    int64_t head_dim = p->head_dim;
    int64_t q_dim = p->n_heads * head_dim;
    int64_t kv_dim = p->n_kv_heads * head_dim;
    int64_t kv_mul = p->n_heads / p->n_kv_heads;
    int64_t hidden_dim = p->hidden_dim;
    float attn_scale = 1.0f / sqrtf((float)p->query_pre_attn_scalar);

    embed_remote_batch(sockfds, tokens, x, dim, p->vocab_size, batch_size);
    float embed_scale = sqrtf((float)dim);
    for (int i = 0; i < batch_size * dim; i++) x[i] *= embed_scale;

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        bool global_layer = is_global_layer(p, l);
        double theta = global_layer ? p->rope_theta_global : p->rope_theta_local;
        double rope_scale = global_layer ? p->rope_scaling_factor : 1.0;

        for (int b = 0; b < batch_size; b++) {
            rmsnorm(s->xb + b * dim, x + b * dim, w->rms_att_weight + l*dim, dim);
        }

        matmul_remote(sockfds, s->qkv, s->xb, dim, q_dim + 2 * kv_dim, l, 8, batch_size);

        int64_t loff = l * steps * kv_dim;
        for (int b = 0; b < batch_size; b++) {
            int pos = pos_start + b;
            float* qkv_b = s->qkv + b * (q_dim + 2 * kv_dim);
            float* q_b = qkv_b;
            float* k_b = s->key_cache + loff + pos * kv_dim;
            float* v_b = s->value_cache + loff + pos * kv_dim;

            memcpy(k_b, qkv_b + q_dim, kv_dim * sizeof(float));
            memcpy(v_b, qkv_b + q_dim + kv_dim, kv_dim * sizeof(float));

            for (int h = 0; h < p->n_heads; h++) {
                norm_and_rotate_head(q_b + h*head_dim, w->rms_q_weight + l*head_dim, head_dim, pos, theta, rope_scale);
            }
            for (int h = 0; h < p->n_kv_heads; h++) {
                norm_and_rotate_head(k_b + h*head_dim, w->rms_k_weight + l*head_dim, head_dim, pos, theta, rope_scale);
            }
        }

        for (int b = 0; b < batch_size; b++) {
            int pos = pos_start + b;
            int64_t window_start = global_layer ? 0 : std::max((int64_t)0, (int64_t)pos - p->sliding_window + 1);
            float* q_b = s->qkv + b * (q_dim + 2 * kv_dim);
            for (int h = 0; h < p->n_heads; h++) {
                float* q_head = q_b + h * head_dim;
                float* att = s->att + h * steps;

                for (int tau = window_start; tau <= pos; tau++) {
                    float* k_head = s->key_cache + loff + tau * kv_dim + (h / kv_mul) * head_dim;
                    float score = 0.0f;
                    for (int i = 0; i < head_dim; i++) {
                        score += q_head[i] * k_head[i];
                    }
                    score *= attn_scale;
                    att[tau] = score;
                }

                softmax(att + window_start, pos - window_start + 1);

                float* xa_head = s->xa + b * q_dim + h * head_dim;
                memset(xa_head, 0, head_dim * sizeof(float));
                for (int tau = window_start; tau <= pos; tau++) {
                    float* v_head = s->value_cache + loff + tau * kv_dim + (h / kv_mul) * head_dim;
                    float a = att[tau];
                    for (int i = 0; i < head_dim; i++) {
                        xa_head[i] += a * v_head[i];
                    }
                }
            }
        }

        matmul_remote(sockfds, s->xb2, s->xa, q_dim, dim, l, 3, batch_size);
        for (int b = 0; b < batch_size; b++) {
            rmsnorm(s->xb2 + b*dim, s->xb2 + b*dim, w->rms_post_att_weight + l*dim, dim);
        }
        for (int i = 0; i < batch_size * dim; i++) {
            x[i] += s->xb2[i];
        }

        for (int b = 0; b < batch_size; b++) {
            rmsnorm(s->xb + b * dim, x + b * dim, w->rms_pre_ffn_weight + l*dim, dim);
        }

        matmul_remote(sockfds, s->hb, s->xb, dim, 2 * hidden_dim, l, 9, batch_size);

        // GeGLU, compacting into s->hb + b*hidden_dim so matmul ID 5 reads a contiguous array
        for (int b = 0; b < batch_size; b++) {
            float* hb_src = s->hb + b * (2 * hidden_dim);
            float* hb2_src = hb_src + hidden_dim;
            float* hb_dst = s->hb + b * hidden_dim;
            for (int i = 0; i < hidden_dim; i++) {
                hb_dst[i] = gelu_tanh(hb_src[i]) * hb2_src[i];
            }
        }

        matmul_remote(sockfds, s->xb, s->hb, hidden_dim, dim, l, 5, batch_size);
        for (int b = 0; b < batch_size; b++) {
            rmsnorm(s->xb + b*dim, s->xb + b*dim, w->rms_post_ffn_weight + l*dim, dim);
        }
        for (int i = 0; i < batch_size * dim; i++) {
            x[i] += s->xb[i];
        }
    }

    float* last_x = x + (batch_size - 1) * dim;
    rmsnorm(last_x, last_x, w->rms_final_weight, dim);

    matmul_remote(sockfds, s->logits, last_x, p->dim, p->vocab_size, 0, 7, 1);

    return s->logits;
}
