#pragma once


#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float *wqkv; // (layer, dim + 2 * n_kv_heads * head_size, dim)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1w3; // (layer, 2 * hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    int64_t dim; // transformer dimension
    int64_t hidden_dim; // for ffn layers
    int64_t n_layers; // number of layers
    int64_t n_heads; // number of query heads
    int64_t n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int64_t vocab_size; // vocabulary size, usually 256 (byte-level)
    int64_t seq_len; // max sequence length
} Config;

inline void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    if (p->dim % p->n_heads != 0) {
        fprintf(stderr, "dim %lld must be divisible by n_heads %lld\n", p->dim, p->n_heads);
        exit(EXIT_FAILURE);
    }
    int64_t head_size = p->dim / p->n_heads;
    int64_t kv_dim = p->n_kv_heads * head_size;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wqkv = ptr;
    ptr += n_layers * (p->dim + 2 * kv_dim) * p->dim;
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1w3 = ptr; 
    ptr += n_layers * p->dim * p->hidden_dim * 2;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
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

inline void dum_client_weights(TransformerWeights *w, Config* p, char* checkpoint_path) {
    FILE *file = fopen(checkpoint_path, "wb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint_path); exit(EXIT_FAILURE); }
    // write the config header
    if (fwrite(p, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // write the weights
    // fwrite(w->token_embedding_table, sizeof(float), p->vocab_size * p->dim, file);
    fwrite(w->rms_att_weight, sizeof(float), p->n_layers * p->dim, file);
    fwrite(w->rms_ffn_weight, sizeof(float), p->n_layers * p->dim, file);
    fwrite(w->rms_final_weight, sizeof(float), p->dim, file);
    fclose(file);
}
