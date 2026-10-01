/* Inference for Llama-2 Transformer model in pure C (Quantized) */

#include "common.hpp"
#include "networking.hpp"
#include "backend/auto.hpp"
#include "backend/cpu.hpp"
#include "constants.hpp"
#include "simdcrypt/PRNG.hpp"

#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <queue>
#include <vector>

int num_layers_in_memory = 8;
bool copy_to_cpu_memory = false;

typedef struct {
    // weights for matmuls. note dim == n_heads * head_size
    integer_w_t *wqkv; // (layer, dim + 2 * n_kv_heads * head_size, dim)
    integer_w_t* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    integer_w_t* w2; // (layer, dim, hidden_dim)
    integer_w_t* w1w3;
    // (optional) classifier weights for the logits, on the last layer
    integer_w_t* wcls;
} TransformerQuantizedWeights;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights floatWeights; // just for the embeddings (float)
    TransformerQuantizedWeights mmapWeights; // mapped directly to the .bin.q file
    TransformerQuantizedWeights cpuWeights; // either points to mmap or pinned host RAM
    TransformerQuantizedWeights *gpuWeights; // rotating weights on GPU
    int num_devices;
    
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

// Helper function to read the custom .bin.q format
int read_quantized_checkpoint(const char* checkpoint, Config* config, TransformerWeights* float_w, TransformerQuantizedWeights* q_w, int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    
    // Read Header
    fread(config, sizeof(Config), 1, file);
    int shared_weights;
    fread(&shared_weights, sizeof(int), 1, file); // This will always be 0 in .bin.q
    
    *fd = fileno(file);
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    
    // Perform mmap BEFORE closing the file descriptor
    *data = (float*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }

    // Now it is safe to close the file. 
    // mmap retains its own internal reference to the inode.
    fclose(file); 
    *fd = -1; // Prevent free_transformer from double-closing the descriptor

    int64_t dim = config->dim;
    int64_t hidden_dim = config->hidden_dim;
    int64_t head_size = dim / config->n_heads;
    int64_t n_layers = config->n_layers;
    int64_t kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;

    char* ptr = (char*)*data + sizeof(Config) + sizeof(int);

    // Map Embeddings (Float)
    float_w->token_embedding_table = new float[config->vocab_size * dim];
    memcpy(float_w->token_embedding_table, ptr, config->vocab_size * dim * sizeof(float));
    // float_w->token_embedding_table = (float*)ptr;
    ptr += config->vocab_size * dim * sizeof(float);

    // Map Quantized Dense Layers (integer_w_t)
    q_w->wqkv = (integer_w_t*)ptr;
    ptr += n_layers * (dim + 2 * kv_dim) * dim * sizeof(integer_w_t);

    q_w->wo = (integer_w_t*)ptr;
    ptr += n_layers * (config->n_heads * head_size) * dim * sizeof(integer_w_t);

    q_w->w1w3 = (integer_w_t*)ptr;
    ptr += n_layers * dim * hidden_dim * 2 * sizeof(integer_w_t);

    q_w->w2 = (integer_w_t*)ptr;
    ptr += n_layers * hidden_dim * dim * sizeof(integer_w_t);

    // Unconditionally map wcls, since shared_weights is 0
    q_w->wcls = (integer_w_t*)ptr;

    return shared_weights;
}

void malloc_and_copy_on_host(Config* p, TransformerQuantizedWeights* w_mmap, TransformerQuantizedWeights* w_cpu, Backend* backend) {
    if (!copy_to_cpu_memory) {
        // Just point directly to the mmap region
        w_cpu->wqkv = w_mmap->wqkv;
        w_cpu->wo   = w_mmap->wo;
        w_cpu->w2   = w_mmap->w2;
        w_cpu->w1w3 = w_mmap->w1w3;
        w_cpu->wcls = w_mmap->wcls;
        return;
    }

    int64_t dim = p->dim;
    int64_t hidden_dim = p->hidden_dim;
    int64_t head_size = dim / p->n_heads;
    int64_t n_layers = p->n_layers;

    size_t size_wqkv = n_layers * (dim + 2 * (p->n_kv_heads * head_size)) * dim;
    size_t size_wo   = n_layers * (p->n_heads * head_size) * dim;
    size_t size_w2   = n_layers * dim * hidden_dim;
    size_t size_w1w3 = n_layers * dim * hidden_dim * 2;
    size_t size_wcls = p->vocab_size * dim;

    // Helper function to allocate OS page-aligned memory
    auto allocate_aligned = [](size_t bytes) -> integer_w_t* {
        void* ptr = nullptr;

        size_t alignment = sysconf(_SC_PAGESIZE); // Usually 4096 bytes
        if (posix_memalign(&ptr, alignment, bytes) != 0) {
            fprintf(stderr, "Failed to allocate aligned memory\n");
            exit(EXIT_FAILURE);
        }
        return static_cast<integer_w_t*>(ptr);
    };

    // Allocate page-aligned host memory
    w_cpu->wqkv = allocate_aligned(size_wqkv * sizeof(integer_w_t));
    w_cpu->wo   = allocate_aligned(size_wo * sizeof(integer_w_t));
    w_cpu->w2   = allocate_aligned(size_w2 * sizeof(integer_w_t));
    w_cpu->w1w3 = allocate_aligned(size_w1w3 * sizeof(integer_w_t));
    w_cpu->wcls = allocate_aligned(size_wcls * sizeof(integer_w_t));

    // Pin memory so the GPU driver can DMA directly without blocking the CPU
    backend->pinMemory(w_cpu->wqkv, size_wqkv * sizeof(integer_w_t));
    backend->pinMemory(w_cpu->wo, size_wo * sizeof(integer_w_t));
    backend->pinMemory(w_cpu->w2, size_w2 * sizeof(integer_w_t));
    backend->pinMemory(w_cpu->w1w3, size_w1w3 * sizeof(integer_w_t));
    backend->pinMemory(w_cpu->wcls, size_wcls * sizeof(integer_w_t));

    printf("Copying pre-quantized weights to pinned CPU memory... ");
    fflush(stdout);
    
    memcpy(w_cpu->wqkv, w_mmap->wqkv, size_wqkv * sizeof(integer_w_t));
    memcpy(w_cpu->wo,   w_mmap->wo,   size_wo * sizeof(integer_w_t));
    memcpy(w_cpu->w2,   w_mmap->w2,   size_w2 * sizeof(integer_w_t));
    memcpy(w_cpu->w1w3, w_mmap->w1w3, size_w1w3 * sizeof(integer_w_t));
    memcpy(w_cpu->wcls, w_mmap->wcls, size_wcls * sizeof(integer_w_t));
    
    printf("Done.\n");
}

void free_cpu_weights(TransformerQuantizedWeights *w_cpu, Backend *backend) {
    if (!copy_to_cpu_memory) {
        return; // Safety guard! Don't unpin/free memory managed by mmap
    }

    backend->unpinMemory(w_cpu->wqkv);
    backend->unpinMemory(w_cpu->wo);
    backend->unpinMemory(w_cpu->w2);
    backend->unpinMemory(w_cpu->w1w3);
    backend->unpinMemory(w_cpu->wcls);

    free(w_cpu->wqkv);
    free(w_cpu->wo);
    free(w_cpu->w2);
    free(w_cpu->w1w3);
    free(w_cpu->wcls);
}

void malloc_on_device(Config* p, TransformerQuantizedWeights *w_gpu, Backend *backend, int num_devices) {
    int64_t dim = p->dim;
    int64_t hidden_dim =  p->hidden_dim;
    int64_t head_size = dim / p->n_heads;

    if (num_layers_in_memory < p->n_layers && num_devices > 1) {
        fprintf(stderr, "num_layers_in_memory (%d) is less than the number of layers in the model (%lld). Cannot use multi-GPU without fully caching on device.\n", num_layers_in_memory, p->n_layers);
        exit(EXIT_FAILURE);
    }

    if (num_layers_in_memory > p->n_layers) {
        fprintf(stderr, "num_layers_in_memory (%d) is greater than the number of layers in the model (%lld). Adjusting to %lld.\n", num_layers_in_memory, p->n_layers, p->n_layers);
        num_layers_in_memory = p->n_layers;
    }

    int layers_per_dev = (num_devices > 1) ? (p->n_layers / num_devices) : num_layers_in_memory;

    // allocate GPU memory
    for (int d = 0; d < num_devices; d++) {
        backend->setDevice(d);
        backend->malloc(&w_gpu[d].wqkv, layers_per_dev * (p->dim + 2 * (p->n_kv_heads * head_size)) * dim * sizeof(integer_w_t));
        backend->malloc(&w_gpu[d].wo, layers_per_dev * (p->n_heads * head_size) * dim * sizeof(integer_w_t));
        backend->malloc(&w_gpu[d].w2, layers_per_dev * dim * hidden_dim * sizeof(integer_w_t));
        backend->malloc(&w_gpu[d].w1w3, layers_per_dev * dim * hidden_dim * 2 * sizeof(integer_w_t));
        
        if (d == 0) {
            backend->malloc(&w_gpu[d].wcls, p->vocab_size * dim * sizeof(integer_w_t));
        }
    }
    backend->setDevice(0);
}

// copy layer `layer` to GPU at index `location` (0, 1, ..., num_layers_in_memory-1)
void copy_to_layer_device(TransformerQuantizedWeights *w_cpu, Config* p, TransformerQuantizedWeights *w_gpu, Backend *backend, int location, int layer, int num_devices) {
    int64_t dim = p->dim;
    int64_t hidden_dim =  p->hidden_dim;
    int64_t head_size = dim / p->n_heads;

    int layers_per_dev = (num_devices > 1) ? (p->n_layers / num_devices) : p->n_layers;
    int dev = (num_devices > 1) ? (layer / layers_per_dev) : 0;
    int local_loc = (num_devices > 1) ? (layer % layers_per_dev) : location;

    backend->setDevice(dev);

    backend->copyHostToDeviceAsync(w_gpu[dev].wqkv + local_loc * (p->dim + 2 * (p->n_kv_heads * head_size))*dim,
        w_cpu->wqkv + layer * (p->dim + 2 * (p->n_kv_heads * head_size))*dim,
        (p->dim + 2 * (p->n_kv_heads * head_size))*dim*sizeof(integer_w_t), layer * 4 + 0);
        
    backend->copyHostToDeviceAsync(w_gpu[dev].wo + local_loc * (p->n_heads*head_size)*dim,
        w_cpu->wo + layer * (p->n_heads*head_size)*dim,
        (p->n_heads*head_size)*dim*sizeof(integer_w_t), layer * 4 + 1);
        
    backend->copyHostToDeviceAsync(w_gpu[dev].w2 + local_loc * dim*hidden_dim,
        w_cpu->w2 + layer * dim*hidden_dim,
        dim*hidden_dim*sizeof(integer_w_t), layer * 4 + 2);
        
    backend->copyHostToDeviceAsync(w_gpu[dev].w1w3 + local_loc * dim*hidden_dim*2,
        w_cpu->w1w3 + layer * dim*hidden_dim*2,
        dim*hidden_dim*2*sizeof(integer_w_t), layer * 4 + 3);
}

void wait_for_layer_copy(Backend *backend, int layer, int num_devices, int n_layers) {
    int layers_per_dev = (num_devices > 1) ? (n_layers / num_devices) : n_layers;
    int dev = (num_devices > 1) ? (layer / layers_per_dev) : 0;
    
    backend->setDevice(dev);
    // Restored synchronization logic
    backend->waitForOperation(layer * 4 + 0);
    backend->waitForOperation(layer * 4 + 1);
    backend->waitForOperation(layer * 4 + 2);
    backend->waitForOperation(layer * 4 + 3);
}

void first_copy_to_device(TransformerQuantizedWeights *w_cpu, Config* p, TransformerQuantizedWeights *w_gpu, Backend *backend, int num_devices) {
    int64_t dim = p->dim;
    backend->setDevice(0);
    // copy wcls in entirety to GPU, since it is not layer-specific
    backend->copyHostToDevice(w_gpu[0].wcls, w_cpu->wcls, p->vocab_size*dim*sizeof(integer_w_t));
    
    int layers_to_copy = (num_devices > 1) ? p->n_layers : num_layers_in_memory;
    // copy layer 0, 1, ... num_layers_in_memory-1 to GPU
    for (int i = 0; i < layers_to_copy; i++) {
        copy_to_layer_device(w_cpu, p, w_gpu, backend, i, i, num_devices);
    }

    for (int i = 0; i < layers_to_copy; i++) {
        wait_for_layer_copy(backend, i, num_devices, p->n_layers);
    }
    backend->setDevice(0);
}

void free_gpu_weights(TransformerQuantizedWeights *w, Backend *backend, int num_devices) {
    // free the GPU memory
    for (int d = 0; d < num_devices; d++) {
        backend->setDevice(d);
        backend->free(w[d].wqkv);
        backend->free(w[d].wo);
        backend->free(w[d].w2);
        backend->free(w[d].w1w3);
        if (d == 0) {
            backend->free(w[d].wcls);
        }
    }
    backend->setDevice(0);
}

void build_transformer(Transformer *t, const char* checkpoint_path, Backend *backend, int num_devices) {
    t->num_devices = num_devices;
    t->gpuWeights = new TransformerQuantizedWeights[num_devices];

    // Read the mixed-type .bin.q checkpoint
    int shared_weights = read_quantized_checkpoint(checkpoint_path, &t->config, &t->floatWeights, &t->mmapWeights, &t->fd, &t->data, &t->file_size);
    
    // Note: We skip dum_client_weights here because it requires a fully floating-point model 
    // to write out to client.bin. You should generate client.bin from the non-quantized server!
    
    // Setup Host Memory (Either mapped or copied to pinned buffers)
    malloc_and_copy_on_host(&t->config, &t->mmapWeights, &t->cpuWeights, backend);
    
    // 2. Setup GPU memory bounds
    malloc_on_device(&t->config, t->gpuWeights, backend, num_devices);
    
    // 3. Initiate DMA transfers
    first_copy_to_device(&t->cpuWeights, &t->config, t->gpuWeights, backend, num_devices);
}

void free_transformer(Transformer* t, Backend *backend) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { 
        munmap(t->data, t->file_size); 
    }
    if (t->fd != -1) { close(t->fd); }
    
    free_cpu_weights(&t->cpuWeights, backend);
    free_gpu_weights(t->gpuWeights, backend, t->num_devices);
    delete[] t->floatWeights.token_embedding_table;
    delete[] t->gpuWeights;
}

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void matmul_server(int sockfd, Transformer* transformer, Backend *backend)
{
    int server_mode;
    read_into(sockfd, server_mode);
    simdcrypt::PRNG prng;
    int steps;
    int prompt_len;
    if (server_mode == 1) { // PRNG optimization mode
        simdcrypt::block seed;
        read_into(sockfd, seed);
        prng.SetSeed(seed);
    }
    read_into(sockfd, steps);
    read_into(sockfd, prompt_len);

    Config* p = &transformer->config;
    TransformerQuantizedWeights* w = transformer->gpuWeights;
    int64_t dim = p->dim;
    int64_t kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int64_t hidden_dim =  p->hidden_dim;

    int actual_prompt_len = std::min(prompt_len, steps);
    if (actual_prompt_len < 1) actual_prompt_len = 1;

    int64_t max_batch = std::max((int64_t)1, (int64_t)actual_prompt_len);
    int64_t x_size = std::max(dim, hidden_dim) * max_batch;
    int64_t max_n = std::max(dim, hidden_dim);
    int64_t max_chunks = (max_n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    // Safely accommodate potential batch_size * num_chunks per return dimension 
    int64_t y_size = std::max({dim, 2 * hidden_dim, kv_dim, p->vocab_size}) * max_chunks * max_batch;
    
    std::vector<integer_t *> d_x(transformer->num_devices);
    std::vector<integer_t *> d_y(transformer->num_devices);
    
    integer_t *h_x = new integer_t[x_size];
    integer_t *h_y = new integer_t[y_size];
    
    if (backend->isCpuBackend()) {
        d_x[0] = h_x;
        d_y[0] = h_y;
    } else {
        for (int d = 0; d < transformer->num_devices; d++) {
            backend->setDevice(d);
            backend->malloc(&d_x[d], x_size * sizeof(integer_t));
            backend->malloc(&d_y[d], y_size * sizeof(integer_t));
        }
        backend->setDevice(0);
    }
    char *one_hot = new char[max_batch * ((p->vocab_size + 7) / 8)];
    float *embedding = new float[max_batch * dim];

    std::queue<int> layers;
    std::queue<int> IDs;
    std::queue<int> batches;

    // 1. Batched Prefill Phase for prompt tokens
    layers.push(0); IDs.push(-1); batches.push(actual_prompt_len); // embedding layer

    for (int j = 0; j < p->n_layers; j++) {
        layers.push(j); IDs.push(8); batches.push(actual_prompt_len);
        layers.push(j); IDs.push(3); batches.push(actual_prompt_len);
        layers.push(j); IDs.push(9); batches.push(actual_prompt_len);
        layers.push(j); IDs.push(5); batches.push(actual_prompt_len);
    }
    layers.push(0); IDs.push(7); batches.push(1); // Classifier for final prompt token

    // 2. Single-token Decode Phase for remaining steps
    for (int i = 0; i < steps - actual_prompt_len; i++) {
        layers.push(0); IDs.push(-1); batches.push(1);
        for (int j = 0; j < p->n_layers; j++) {
            layers.push(j); IDs.push(8); batches.push(1);
            layers.push(j); IDs.push(3); batches.push(1);
            layers.push(j); IDs.push(9); batches.push(1);
            layers.push(j); IDs.push(5); batches.push(1);
        }
        layers.push(0); IDs.push(7); batches.push(1);
    }
    layers.push(-1);

    auto pop_front = [](std::queue<int>& q) {
        int front = q.front();
        q.pop();
        return front;
    };

    int current_layer_location = 0;
    int layers_per_dev = (transformer->num_devices > 1) ? (p->n_layers / transformer->num_devices) : p->n_layers;

    int64_t total_comm_time = 0;
    int64_t total_emb_comp_time = 0;
    int64_t total_mv_comp_time = 0;
    int64_t total_prg_time = 0;
    int64_t total_layer_copy_time = 0;
    int64_t total_input_copy_time = 0;

    while (true)
    {
        int layer, ID, batch_size;
        layer = pop_front(layers);

        if (layer == -1) { break; }

        ID = pop_front(IDs);
        batch_size = pop_front(batches);

        integer_w_t *d_w;
        int64_t n, d;

        int dev = 0;
        if (ID != -1 && ID != 7 && transformer->num_devices > 1) {
            dev = layer / layers_per_dev;
        }
        backend->setDevice(dev);
        
        int local_layer_loc = (transformer->num_devices > 1) ? (layer % layers_per_dev) : current_layer_location;

        switch(ID)
        {
            case 3: // wo
                n = dim;
                d = dim;
                d_w = w[dev].wo + local_layer_loc * n * d;
                break;
            case 5: // w2
                n = hidden_dim;
                d = dim;
                d_w = w[dev].w2 + local_layer_loc * n * d;
                break;
            case 7: // wcls
                n = dim;
                d = p->vocab_size;
                d_w = w[0].wcls;
                break;
            case 8: // qkv
                n = dim;
                d = dim + 2 * kv_dim;
                d_w = w[dev].wqkv + local_layer_loc * n * d;
                break;
            case 9: // w1 | w3
                n = dim;
                d = 2 * hidden_dim;
                d_w = w[dev].w1w3 + local_layer_loc * n * d;
                break;
            case -1:
                break;
            default:
                fprintf(stderr, "Unknown matmul ID %d\n", ID);
                exit(EXIT_FAILURE);
        }
        
        // read x
        if (ID == -1) { // embedding layer
            if (server_mode == 0) {
                int64_t start_time = time_in_ms();
                read_into(sockfd, one_hot, batch_size * ((p->vocab_size + 7) / 8));
                total_comm_time += (time_in_ms() - start_time);
            } else {
                int64_t start_time = time_in_ms();
                prng.get(one_hot, batch_size * ((p->vocab_size + 7) / 8));
                total_prg_time += (time_in_ms() - start_time);
            }
            int64_t start_time = time_in_ms();
            memset(embedding, 0, batch_size * dim * sizeof(float));
            // Process one-hot encoded vector to get embedding
            for (int b = 0; b < batch_size; b++) {
                for (int i = 0; i < p->vocab_size; i++) {
                    if (one_hot[b * ((p->vocab_size + 7) / 8) + i / 8] & (1 << (i % 8))) {
                        #pragma omp simd
                        for (int j = 0; j < dim; j++) {
                            ((uint32_t *)embedding)[b * dim + j] ^= ((uint32_t *)transformer->floatWeights.token_embedding_table)[i * dim + j];
                        }
                    }
                }
            }
            total_emb_comp_time += (time_in_ms() - start_time);
            start_time = time_in_ms();
            write_into(sockfd, (char *)embedding, batch_size * dim * sizeof(float));
            total_comm_time += (time_in_ms() - start_time);
        }
        else {
            if (ID == 8 && num_layers_in_memory < p->n_layers && transformer->num_devices == 1) {
                // wait for the layer copy to finish before using it
                int64_t start_time = time_in_ms();
                wait_for_layer_copy(backend, layer, transformer->num_devices, p->n_layers);
                total_layer_copy_time += (time_in_ms() - start_time);
            }
            
            int64_t n_total = n * batch_size;
            if (server_mode == 0) {
                int64_t start_time = time_in_ms();
                read_into(sockfd, (char *)h_x, n_total * sizeof(integer_t));
                total_comm_time += (time_in_ms() - start_time);
            } else {
                int64_t start_time = time_in_ms();
                prng.get(h_x, n_total);
                total_prg_time += (time_in_ms() - start_time);
            }
            
            // copy x to GPU
            if (!backend->isCpuBackend()) {
                int64_t start_time = time_in_ms();
                backend->copyHostToDevice(d_x[dev], h_x, n_total * sizeof(integer_t));
                total_input_copy_time += (time_in_ms() - start_time);
            }
            
            // do the matmul
            int num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
            int64_t start_time = time_in_ms();
            backend->matmul(d_y[dev], d_x[dev], d_w, n, d, batch_size);
            total_mv_comp_time += (time_in_ms() - start_time);

            // Copy the (still per-chunk) matmul result back to host memory
            int64_t chunked_total = d * num_chunks * batch_size;
            if (!backend->isCpuBackend()) {
                int64_t start_time = time_in_ms();
                backend->copyDeviceToHost(h_y, d_y[dev], chunked_total * sizeof(integer_t));
                total_input_copy_time += (time_in_ms() - start_time);
            }

            // Accumulate the per-chunk partial sums here on the server CPU so the
            // client receives one value per output element instead of one per chunk.
            // Layout: h_y[b * num_chunks * d + c * d + i] -> h_y[b * d + i].
            // Reducing in place is safe: each destination range [b*d, (b+1)*d) sits
            // at or before that batch's own source region for num_chunks >= 1.
            if (num_chunks > 1) {
                start_time = time_in_ms();
                for (int b = 0; b < batch_size; b++) {
                    const integer_t *src = h_y + (int64_t)b * num_chunks * d;
                    integer_t *dst = h_y + (int64_t)b * d;
                    for (int i = 0; i < d; i++) {
                        integer_t acc = 0;
                        for (int c = 0; c < num_chunks; c++) {
                            acc += src[(int64_t)c * d + i];
                        }
                        dst[i] = acc;
                    }
                }
                total_mv_comp_time += (time_in_ms() - start_time);
            }

            int64_t d_total = d * batch_size;
            write_into(sockfd, (char *)h_y, d_total * sizeof(integer_t));
        }

        if (ID == 5 && transformer->num_devices == 1) {
            // layer is used, replace
            // copy layer `layer + max_layers_in_memory` to GPU at index `current_layer_location`
            if (num_layers_in_memory < p->n_layers) {
                int next_layer = (layer + num_layers_in_memory) % p->n_layers;
                // Pull directly from CPU memory now
                int64_t start_time = time_in_ms();
                copy_to_layer_device(&transformer->cpuWeights, p, transformer->gpuWeights, backend, current_layer_location, next_layer, transformer->num_devices);
                total_layer_copy_time += (time_in_ms() - start_time);
            }
            current_layer_location = (current_layer_location + 1) % num_layers_in_memory;
        }
    }

    // Fixed: properly delete arrays allocated with new[]
    delete[] h_x;
    delete[] h_y;
    delete[] one_hot;
    delete[] embedding;

    printf("Server finished processing %d steps. Stats:\n", steps);
    printf("    Total communication time: %lld ms\n", total_comm_time);
    printf("    Total embedding computation time: %lld ms\n", total_emb_comp_time);
    printf("    Total matrix-vector computation time: %lld ms\n", total_mv_comp_time);
    printf("    Total PRNG time: %lld ms\n", total_prg_time);
    printf("    Total layer copy time: %lld ms\n", total_layer_copy_time);
    printf("    Total input copy time: %lld ms\n", total_input_copy_time);

    if (!backend->isCpuBackend()) {
        for (int d = 0; d < transformer->num_devices; d++) {
            backend->setDevice(d);
            backend->free(d_x[d]);
            backend->free(d_y[d]);
        }
        backend->setDevice(0);
    }

    // RESET GPU LAYER STATE FOR EVERY NEW CONNECTION
    if (num_layers_in_memory < p->n_layers && transformer->num_devices == 1) {
        for (int i = 0; i < num_layers_in_memory; i++) {
            if (i < p->n_layers) {
                copy_to_layer_device(&transformer->cpuWeights, p, transformer->gpuWeights, backend, i, i, transformer->num_devices);
            }
        }
        for (int i = 0; i < num_layers_in_memory; i++) {
            wait_for_layer_copy(backend, i, transformer->num_devices, p->n_layers);
        }
    }
}

void error_usage() {
    fprintf(stderr, "Usage:   server <model_name>\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *model_name = NULL;
    char *tokenizer_path = (char*)"tokenizer.bin";
    int port = DEFAULT_PORT;

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { model_name = argv[1]; } else { error_usage(); }
    bool force_cpu = false;
    int num_devices = 1;
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--force-cpu") == 0) {
            force_cpu = true;
        } else if (strcmp(argv[i], "--copy-to-cpu") == 0) {
            copy_to_cpu_memory = true;
        } else if (strcmp(argv[i], "--device-layers") == 0) {
            num_layers_in_memory = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--port") == 0) {
            port = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--num-devices") == 0) {
            num_devices = atoi(argv[i + 1]);
            i++;
        } else {
            error_usage();
        }
    }

    if (force_cpu && copy_to_cpu_memory) {
        printf("Warning: Using `copy-to-cpu` with `force-cpu` achieves no benefit and only causes extra memory usage.\n");
        copy_to_cpu_memory = false;
    }

    Backend *backend;

    if (force_cpu) {
        backend = new CpuBackend();
    } else {
        backend = new AutoBackend();
    }
    
    backend->initDevices(num_devices);
    num_devices = backend->getNumDevices();
    
    std::cout << "Using: " << backend->getDeviceName() << " across " << num_devices << " devices." << std::endl;
    std::cout << "Memory strategy: " << (copy_to_cpu_memory ? "Pinned CPU RAM (Fast async)" : "Direct MMAP (Low RAM)") << std::endl;

    // build the Transformer via the model .bin file
    Transformer transformer;
    std::string checkpoint_path = std::string(getenv("HOME")) + std::string("/.dvllm/") + model_name + std::string("/model.bin.q");
    build_transformer(&transformer, checkpoint_path.c_str(), backend, num_devices);
    std::cout << "Transformer model loaded from " << checkpoint_path << std::endl;


    // start the server
    loop([&](int sockfd) {
        matmul_server(sockfd, &transformer, backend);
    }, port);

    free_transformer(&transformer, backend);

    delete backend;

    return 0;
}