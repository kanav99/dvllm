#pragma once

// Gemma 3 analogue of common.hpp. Kept separate from common.hpp because the
// architecture differs enough (decoupled head_dim, QK-norm, sandwich norm,
// dual RoPE theta / sliding window) that trying to share one Config/
// TransformerWeights struct with Llama would make both harder to read.

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>

typedef struct {
    int64_t dim;             // transformer (residual stream) dimension
    int64_t hidden_dim;      // for ffn layers
    int64_t n_layers;        // number of layers
    int64_t n_heads;         // number of query heads
    int64_t n_kv_heads;      // number of key/value heads
    int64_t vocab_size;      // vocabulary size
    int64_t seq_len;         // max sequence length
    int64_t head_dim;        // per-head dim; NOT necessarily dim / n_heads for Gemma 3
    int64_t sliding_window;  // local attention window size
    int64_t sliding_window_pattern; // every Nth layer (1-indexed) is global attention
    double query_pre_attn_scalar;   // attention score scale = 1/sqrt(this)
    double rope_theta_local;        // RoPE base freq for local (sliding window) layers
    double rope_theta_global;       // RoPE base freq for global layers
    double rope_scaling_factor;     // linear RoPE scaling applied to global layers only (1.0 = none)
} Config;

inline bool is_global_layer(const Config* p, int64_t layer) {
    return ((layer + 1) % p->sliding_window_pattern) == 0;
}

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // sandwich-norm weights (layer, dim), already baked with +1.0 at export time
    float* rms_att_weight;       // input_layernorm
    float* rms_post_att_weight;  // post_attention_layernorm
    float* rms_pre_ffn_weight;   // pre_feedforward_layernorm
    float* rms_post_ffn_weight;  // post_feedforward_layernorm
    // QK-norm weights (layer, head_dim)
    float* rms_q_weight;
    float* rms_k_weight;
    // weights for matmuls. note n_heads*head_dim (== q_dim) need not equal dim
    float* wqkv; // (layer, q_dim + 2*kv_dim, dim)
    float* wo;   // (layer, dim, q_dim)
    // weights for ffn
    float* w1w3; // (layer, 2*hidden_dim, dim)
    float* w2;   // (layer, dim, hidden_dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // classifier weights for the logits (always == token_embedding_table for Gemma 3, tied)
    float* wcls;
} TransformerWeights;

inline void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int64_t q_dim = p->n_heads * p->head_dim;
    int64_t kv_dim = p->n_kv_heads * p->head_dim;
    unsigned long long n_layers = p->n_layers;

    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;

    w->rms_att_weight = ptr;      ptr += n_layers * p->dim;
    w->rms_post_att_weight = ptr; ptr += n_layers * p->dim;
    w->rms_pre_ffn_weight = ptr;  ptr += n_layers * p->dim;
    w->rms_post_ffn_weight = ptr; ptr += n_layers * p->dim;
    w->rms_q_weight = ptr;        ptr += n_layers * p->head_dim;
    w->rms_k_weight = ptr;        ptr += n_layers * p->head_dim;

    w->wqkv = ptr;
    ptr += n_layers * (q_dim + 2 * kv_dim) * p->dim;
    w->wo = ptr;
    ptr += n_layers * p->dim * q_dim;
    w->w1w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim * 2;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;

    w->rms_final_weight = ptr;
    ptr += p->dim;

    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

inline void read_checkpoint(const char* checkpoint, Config* config, TransformerWeights* weights,
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

inline void dum_client_weights(TransformerWeights *w, Config* p, char* checkpoint_path) {
    FILE *file = fopen(checkpoint_path, "wb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint_path); exit(EXIT_FAILURE); }
    if (fwrite(p, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    int shared_weights = 1; // client.bin's header shape must match model.bin/model.bin.q (Config + int32)
    fwrite(&shared_weights, sizeof(int), 1, file);
    fwrite(w->rms_att_weight, sizeof(float), p->n_layers * p->dim, file);
    fwrite(w->rms_post_att_weight, sizeof(float), p->n_layers * p->dim, file);
    fwrite(w->rms_pre_ffn_weight, sizeof(float), p->n_layers * p->dim, file);
    fwrite(w->rms_post_ffn_weight, sizeof(float), p->n_layers * p->dim, file);
    fwrite(w->rms_q_weight, sizeof(float), p->n_layers * p->head_dim, file);
    fwrite(w->rms_k_weight, sizeof(float), p->n_layers * p->head_dim, file);
    fwrite(w->rms_final_weight, sizeof(float), p->dim, file);
    fclose(file);
}
