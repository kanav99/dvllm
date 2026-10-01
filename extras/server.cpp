/* Inference for Llama-2 Transformer model in pure C */

#include "common.hpp"
#include "networking.hpp"
#include "backend/auto.hpp"
#include "backend/cpu.hpp"

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

int num_layers_in_memory = 8;
bool copy_to_cpu_memory = false;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the original float weights of the model (mmap)
    TransformerWeights cpuWeights; // pinned host RAM for async transfers
    TransformerWeights gpuWeights; // rotating weights on GPU
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_and_copy_on_host(Config* p, TransformerWeights* w, TransformerWeights* w_cpu, Backend* backend) {
    if (!copy_to_cpu_memory) {
        w_cpu->wqkv = w->wqkv;
        w_cpu->wo = w->wo;
        w_cpu->w2 = w->w2;
        w_cpu->w1w3 = w->w1w3;
        w_cpu->wcls = w->wcls;
        return;
    }

    int64_t dim = p->dim;
    int64_t hidden_dim = p->hidden_dim;
    int64_t head_size = dim / p->n_heads;
    int64_t n_layers = p->n_layers;

    size_t size_wqkv = n_layers * (dim + 2 * (p->n_kv_heads * head_size)) * dim;
    size_t size_wo = n_layers * (p->n_heads * head_size) * dim;
    size_t size_w2 = n_layers * dim * hidden_dim;
    size_t size_w1w3 = n_layers * dim * hidden_dim * 2;
    size_t size_wcls = p->vocab_size * dim;

    auto allocate_aligned = [](size_t bytes) -> float* {
        void* ptr = nullptr;

        size_t alignment = sysconf(_SC_PAGESIZE); // Usually 4096 bytes
        if (posix_memalign(&ptr, alignment, bytes) != 0) {
            fprintf(stderr, "Failed to allocate aligned memory\n");
            exit(EXIT_FAILURE);
        }
        return static_cast<float*>(ptr);
    };

    // Allocate host memory using the backend's native pinned allocator
    w_cpu->wqkv = allocate_aligned(size_wqkv * sizeof(float));
    w_cpu->wo = allocate_aligned(size_wo * sizeof(float));
    w_cpu->w2 = allocate_aligned(size_w2 * sizeof(float));
    w_cpu->w1w3 = allocate_aligned(size_w1w3 * sizeof(float));
    w_cpu->wcls = allocate_aligned(size_wcls * sizeof(float));

    backend->pinMemory(w_cpu->wqkv, size_wqkv * sizeof(float));
    backend->pinMemory(w_cpu->wo, size_wo * sizeof(float));
    backend->pinMemory(w_cpu->w2, size_w2 * sizeof(float));
    backend->pinMemory(w_cpu->w1w3, size_w1w3 * sizeof(float));
    backend->pinMemory(w_cpu->wcls, size_wcls * sizeof(float));

    printf("Copying weights to pinned CPU memory... ");
    fflush(stdout);
    
    // Copy from the mmap region to the pinned host buffers
    memcpy(w_cpu->wqkv, w->wqkv, size_wqkv * sizeof(float));
    memcpy(w_cpu->wo, w->wo, size_wo * sizeof(float));
    memcpy(w_cpu->w2, w->w2, size_w2 * sizeof(float));
    memcpy(w_cpu->w1w3, w->w1w3, size_w1w3 * sizeof(float));
    memcpy(w_cpu->wcls, w->wcls, size_wcls * sizeof(float));
    
    printf("Done.\n");
}

void free_cpu_weights(TransformerWeights *w_cpu, Backend *backend) {
    if (copy_to_cpu_memory) {
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
}

void malloc_on_device(Config* p, TransformerWeights *w_gpu, Backend *backend) {
    int64_t dim = p->dim;
    int64_t hidden_dim =  p->hidden_dim;
    int64_t head_size = dim / p->n_heads;

    if (num_layers_in_memory > p->n_layers) {
        fprintf(stderr, "num_layers_in_memory (%d) is greater than the number of layers in the model (%lld). Adjusting to %lld.\n", num_layers_in_memory, p->n_layers, p->n_layers);
        num_layers_in_memory = p->n_layers;
    }
    // allocate GPU memory
    backend->malloc(&w_gpu->wqkv, num_layers_in_memory * (p->dim + 2 * (p->n_kv_heads * head_size)) * dim * sizeof(float));
    backend->malloc(&w_gpu->wo, num_layers_in_memory * (p->n_heads * head_size) * dim * sizeof(float));
    backend->malloc(&w_gpu->w2, num_layers_in_memory * dim * hidden_dim * sizeof(float));
    backend->malloc(&w_gpu->w1w3, num_layers_in_memory * dim * hidden_dim * 2 * sizeof(float));
    backend->malloc(&w_gpu->wcls, p->vocab_size * dim * sizeof(float));
}

// copy layer `layer` to GPU at index `location` (0, 1, ..., num_layers_in_memory-1)
void copy_to_layer_device(TransformerWeights *w_cpu, Config* p, TransformerWeights *w_gpu, Backend *backend, int location, int layer) {
    int64_t dim = p->dim;
    int64_t hidden_dim =  p->hidden_dim;
    int64_t head_size = dim / p->n_heads;

    // copy weights to GPU using async streams from pinned host buffers
    backend->copyHostToDeviceAsync(w_gpu->wqkv + location * (p->dim + 2 * (p->n_kv_heads * head_size))*dim,
        w_cpu->wqkv + layer * (p->dim + 2 * (p->n_kv_heads * head_size))*dim,
        (p->dim + 2 * (p->n_kv_heads * head_size))*dim*sizeof(float), layer * 4 + 0);
        
    backend->copyHostToDeviceAsync(w_gpu->wo + location * (p->n_heads*head_size)*dim,
        w_cpu->wo + layer * (p->n_heads*head_size)*dim,
        (p->n_heads*head_size)*dim*sizeof(float), layer * 4 + 1);
        
    backend->copyHostToDeviceAsync(w_gpu->w2 + location * dim*hidden_dim,
        w_cpu->w2 + layer * dim*hidden_dim,
        dim*hidden_dim*sizeof(float), layer * 4 + 2);
        
    backend->copyHostToDeviceAsync(w_gpu->w1w3 + location * dim*hidden_dim*2,
        w_cpu->w1w3 + layer * dim*hidden_dim*2,
        dim*hidden_dim*2*sizeof(float), layer * 4 + 3);
}

void wait_for_layer_copy(Backend *backend, int layer) {
    backend->waitForOperation(layer * 4 + 0);
    backend->waitForOperation(layer * 4 + 1);
    backend->waitForOperation(layer * 4 + 2);
    backend->waitForOperation(layer * 4 + 3);
}

void first_copy_to_device(TransformerWeights *w_cpu, Config* p, TransformerWeights *w_gpu, Backend *backend) {
    int64_t dim = p->dim;

    // copy wcls in entirety to GPU, since it is not layer-specific
    backend->copyHostToDevice(w_gpu->wcls, w_cpu->wcls, p->vocab_size*dim*sizeof(float));
    
    // copy layer 0, 1, ... num_layers_in_memory-1 to GPU
    for (int i = 0; i < num_layers_in_memory; i++) {
        copy_to_layer_device(w_cpu, p, w_gpu, backend, i, i);
    }
}

void free_gpu_weights(TransformerWeights *w, Backend *backend) {
    // free the GPU memory
    backend->free(w->wqkv);
    backend->free(w->wo);
    backend->free(w->w2);
    backend->free(w->w1w3);
    backend->free(w->wcls); 
}

void build_transformer(Transformer *t, const char* checkpoint_path, Backend *backend) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // dum_client_weights(&t->weights, &t->config, "client.bin"); // now handled by quantize.cpp
    
    // Setup pinned Host Memory
    malloc_and_copy_on_host(&t->config, &t->weights, &t->cpuWeights, backend);
    
    // Setup GPU memory bounds
    malloc_on_device(&t->config, &t->gpuWeights, backend);
    
    // Initiate DMA transfers
    first_copy_to_device(&t->cpuWeights, &t->config, &t->gpuWeights, backend);
}

void free_transformer(Transformer* t, Backend *backend) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { 
        munmap(t->data, t->file_size); 
    }
    if (t->fd != -1) { close(t->fd); }
    
    free_cpu_weights(&t->cpuWeights, backend);
    free_gpu_weights(&t->gpuWeights, backend);
}

void matmul_server(int sockfd, Transformer* transformer, Backend *backend)
{
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->gpuWeights;
    int64_t dim = p->dim;
    int64_t kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int64_t kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int64_t hidden_dim =  p->hidden_dim;
    int64_t head_size = dim / p->n_heads;

    int64_t x_size = std::max(dim, hidden_dim);
    int64_t y_size = std::max(std::max(dim, hidden_dim), std::max(kv_dim, p->vocab_size));
    float *x = new float[x_size];
    float *y = new float[y_size];
    float *d_x, *d_y;
    backend->malloc(&d_x, x_size * sizeof(float));
    backend->malloc(&d_y, y_size * sizeof(float));

    int current_layer_location = 0;
    while (true)
    {
        int layer, ID;
        read_into(sockfd, layer);
        if (layer == -1) { break; }

        read_into(sockfd, ID);
        
        float *d_w;
        int64_t n, d;

        switch(ID)
        {
            case 3: // wo
                n = dim;
                d = dim;
                d_w = w->wo + current_layer_location * n * d;
                break;
            case 5: // w2
                n = hidden_dim;
                d = dim;
                d_w = w->w2 + current_layer_location * n * d;
                break;
            case 7: // wcls
                n = dim;
                d = p->vocab_size;
                d_w = w->wcls;
                break;
            case 8: // qkv
                n = dim;
                d = dim + 2 * kv_dim;
                d_w = w->wqkv + current_layer_location * n * d;
                break;
            case 9: // w1 | w3
                n = dim;
                d = 2 * hidden_dim;
                d_w = w->w1w3 + current_layer_location * n * d;
                break;
            case -1:
                break;
            default:
                fprintf(stderr, "Unknown matmul ID %d\n", ID);
                exit(EXIT_FAILURE);
        }
        // read x
        if (ID == -1) { // embedding layer
            int token;
            read_into(sockfd, token);
            float *embedding = transformer->weights.token_embedding_table + token * dim;
            write_into(sockfd, (char *)embedding, dim * sizeof(float));
        }
        else {
            if (ID == 8) {
                // wait for the layer copy to finish before using it
                if (num_layers_in_memory < p->n_layers) {
                    wait_for_layer_copy(backend, layer);
                }
            }
            read_into(sockfd, (char *)x, n * sizeof(float));
            // copy x to GPU
            backend->copyHostToDevice(d_x, x, n * sizeof(float));
            // do the matmul
            backend->matmul(d_y, d_x, d_w, n, d);
            // copy y back to CPU
            backend->copyDeviceToHost(y, d_y, d * sizeof(float));
            write_into(sockfd, (char *)y, d * sizeof(float));
        }

        if (ID == 5) {
            // layer is used, replace
            // copy layer `layer + max_layers_in_memory` to GPU at index `current_layer_location`
            if (num_layers_in_memory < p->n_layers) {
                int next_layer = (layer + num_layers_in_memory) % p->n_layers;
                copy_to_layer_device(&transformer->cpuWeights, p, &transformer->gpuWeights, backend, current_layer_location, next_layer);
            }
            current_layer_location = (current_layer_location + 1) % num_layers_in_memory;
        }
    }
    // free x and y
    delete[] x;
    delete[] y;
    backend->free(d_x);
    backend->free(d_y);

    if (num_layers_in_memory < p->n_layers) {
        for (int i = 0; i < num_layers_in_memory; i++) {
            if (i < p->n_layers) {
                copy_to_layer_device(&transformer->cpuWeights, p, &transformer->gpuWeights, backend, i, i);
            }
        }
        for (int i = 0; i < num_layers_in_memory; i++) {
            wait_for_layer_copy(backend, i);
        }
    }
}

void error_usage() {
    fprintf(stderr, "Usage:   server <model_name> [options]\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *model_name = NULL;  // e.g. meta-llama/Llama-2-7b
    char *tokenizer_path = "tokenizer.bin";
    int port = DEFAULT_PORT;

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { model_name = argv[1]; } else { error_usage(); }
    bool force_cpu = false;
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
    }
    else {
        backend = new AutoBackend();
    }
    std::cout << "Using: " << backend->getDeviceName() << std::endl;
    std::cout << "Memory strategy: " << (copy_to_cpu_memory ? "Pinned CPU RAM (Fast async)" : "Direct MMAP (Low RAM)") << std::endl;

    // build the Transformer via the model .bin file
    Transformer transformer;
    // fetch model from "~/.dvllm/{model_name}/model.bin"
    std::string checkpoint_path = std::string(getenv("HOME")) + std::string("/.dvllm/") + model_name + std::string("/model.bin");
    build_transformer(&transformer, checkpoint_path.c_str(), backend);
    std::cout << "Transformer model loaded from " << checkpoint_path << std::endl;


    // start the server
    loop([&](int sockfd) {
        matmul_server(sockfd, &transformer, backend);
    }, port);

    free_transformer(&transformer, backend);

    delete backend;

    return 0;
}
