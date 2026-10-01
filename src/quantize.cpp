#include "common.hpp"
#include "constants.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <string>
#include "verification.hpp"
#include <filesystem>
#include "simdcrypt/SHA256.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    // Updated to require 3 arguments to handle both server and client splits
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_name> [--no-clobber] [--no-commit] [--embed-only]\n"
                  << "  --embed-only   only regenerate the embedding commitments (embeddings.bin),\n"
                  << "                 reusing the existing model.bin.q and weight commitments.\n";
        return EXIT_FAILURE;
    }

    std::string input_path = std::string(getenv("HOME")) + std::string("/.dvllm/") + std::string(argv[1]) + std::string("/model.bin");
    std::string server_output_path = std::string(getenv("HOME")) + std::string("/.dvllm/") + std::string(argv[1]) + std::string("/model.bin.q");
    std::string client_output_path = std::string(getenv("HOME")) + std::string("/.dvllm/") + std::string(argv[1]) + std::string("/client.bin");

    bool no_clobber = false;
    bool no_commit = false;
    bool embed_only = false; // only regenerate embeddings.bin, reuse everything else
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-clobber") {
            no_clobber = true;
        } else if (arg == "--no-commit") {
            no_commit = true;
        } else if (arg == "--embed-only") {
            embed_only = true;
        }
    }

    Config config;
    TransformerWeights float_weights;
    int fd = -1;
    float* data = (float*)MAP_FAILED;
    ssize_t file_size = 0;
    std::vector<float> embed_storage; // backing store for embeddings read from model.bin.q

    if (embed_only) {
        // Read the config + float embeddings straight from the quantized model so we
        // never touch model.bin (which is typically purged). The quantizer stores the
        // token embeddings as raw floats right after the header, so the hashes computed
        // here are bit-identical to those from a full run over model.bin.
        std::cout << "--embed-only: reading embeddings from " << server_output_path << "...\n";
        FILE* qf = fopen(server_output_path.c_str(), "rb");
        if (!qf) {
            std::cerr << "Failed to open " << server_output_path << " for reading.\n";
            return EXIT_FAILURE;
        }
        int shared_weights_q = 0;
        if (fread(&config, sizeof(Config), 1, qf) != 1 ||
            fread(&shared_weights_q, sizeof(int), 1, qf) != 1) {
            std::cerr << "Failed to read header from " << server_output_path << "\n";
            fclose(qf);
            return EXIT_FAILURE;
        }
        size_t size_embed = (size_t)config.vocab_size * config.dim;
        embed_storage.resize(size_embed);
        if (fread(embed_storage.data(), sizeof(float), size_embed, qf) != size_embed) {
            std::cerr << "Failed to read embeddings from " << server_output_path << "\n";
            fclose(qf);
            return EXIT_FAILURE;
        }
        fclose(qf);
        float_weights.token_embedding_table = embed_storage.data();
    } else {
        std::cout << "Loading original float model: " << input_path << "...\n";
        // Load using your existing common.hpp parser
        read_checkpoint((char*)input_path.c_str(), &config, &float_weights, &fd, &data, &file_size);
    }

    // --- NEW: Dump the client weights offline ---
    if (embed_only) {
        std::cout << "--embed-only: skipping client weights.\n";
    } else if (no_clobber && fs::exists(client_output_path)) {
        std::cerr << "Client output file already exists, not regenreating \n";
    } else {
        std::cout << "Dumping client weights to " << client_output_path << "...\n";
        dum_client_weights(&float_weights, &config, (char*)client_output_path.c_str());
    }

    if (embed_only) {
        std::cout << "--embed-only: skipping server weights (reusing existing model.bin.q).\n";
    } else if (no_clobber && fs::exists(server_output_path)) {
        std::cerr << "Server output file already exists, not regenreating \n";
    } else {
        std::cout << "Opening output file for server weights...\n";
        FILE* out = fopen(server_output_path.c_str(), "wb");
        if (!out) {
            std::cerr << "Failed to open " << server_output_path << " for writing.\n";
            return EXIT_FAILURE;
        }

        // 1. Write Header
        // We write the config struct followed by 0 for shared_weights, as we are decoupling wcls!
        fwrite(&config, sizeof(Config), 1, out);
        int export_shared_weights = 0; 
        fwrite(&export_shared_weights, sizeof(int), 1, out);

        int64_t dim = config.dim;
        int64_t hidden_dim = config.hidden_dim;
        int64_t head_size = dim / config.n_heads;
        int64_t n_layers = config.n_layers;
        int64_t kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;

        // Size calculations
        size_t size_embed = config.vocab_size * dim;
        size_t size_wqkv  = n_layers * (dim + 2 * kv_dim) * dim;
        size_t size_wo    = n_layers * (config.n_heads * head_size) * dim;
        size_t size_w2    = n_layers * dim * hidden_dim;
        size_t size_w1w3  = n_layers * dim * hidden_dim * 2;
        size_t size_wcls  = config.vocab_size * dim; // Always calculate this now

        // Helper to scale and write integer arrays
        auto write_quantized = [&](float* ptr, size_t elements, const char* name) {
            std::cout << "Quantizing " << name << " (" << elements << " elements)...\n";
            std::vector<integer_w_t> buffer(elements);
            for (size_t i = 0; i < elements; i++) {
                buffer[i] = static_cast<integer_w_t>(ptr[i] * FP_W_SCALE);
            }
            fwrite(buffer.data(), sizeof(integer_w_t), elements, out);
        };

        // 2. Write Embeddings (kept as floats for the client-socket pass-through)
        std::cout << "Writing float embeddings...\n";
        fwrite(float_weights.token_embedding_table, sizeof(float), size_embed, out);

        // 3. Quantize and Write Dense Matrices
        write_quantized(float_weights.wqkv, size_wqkv, "wqkv");
        write_quantized(float_weights.wo, size_wo, "wo");
        write_quantized(float_weights.w1w3, size_w1w3, "w1w3");
        write_quantized(float_weights.w2, size_w2, "w2");
        
        // Always explicitly quantize and write wcls. 
        // If original was shared, float_weights.wcls inherently points to embeddings in memory.
        write_quantized(float_weights.wcls, size_wcls, "wcls");

        fclose(out);
        
        // // Cleanup the mmap from the float file
        // if (data != MAP_FAILED) { munmap(data, file_size); }
        // if (fd != -1) { close(fd); }
        std::cout << "Success! Saved quantized server model to " << server_output_path << "\n";
    }

    if (!no_commit || embed_only) {
        std::cout << "Beginning MCS stuff..." << std::endl;
        std::string commit_directory = std::string(getenv("HOME")) + std::string("/.dvllm/") + std::string(argv[1]) + std::string("/commits/");
        // if commit directory doesn't exist, create it
        try {
            // Creates the directory and any missing parent directories
            // Does nothing and returns false if the directory already exists
            if (fs::create_directories(commit_directory)) {
                std::cout << "Commit directory created successfully.\n";
            } else {
                std::cout << "Commit directory already exists.\n";
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error: " << e.what() << '\n';
        }

        std::cout << "Committing embeddings..." << std::endl;
        std::string embed_path = commit_directory + std::string("embeddings.bin");
        std::ofstream embed_file(embed_path, std::ios::binary);

        simdcrypt::SHA256 hasher;
        uint8_t hash[simdcrypt::SHA256::HashSize];

        for (int i = 0; i < config.vocab_size; ++i) {
            // hasher.Update((uint8_t *)(float_weights.token_embedding_table + i * config.dim), config.dim * sizeof(float));
            // hasher.Final(hash);
            hasher.Hash((uint8_t *)(float_weights.token_embedding_table + i * config.dim), config.dim * sizeof(float), hash);
            embed_file.write(reinterpret_cast<char*>(hash), simdcrypt::SHA256::HashSize);
        }
        embed_file.close();
        std::cout << "Wrote embedding commitments to " << embed_path << "\n";

        if (embed_only) {
            std::cout << "--embed-only: skipping setup + weight commitments.\n";
        } else {
            std::string setup_path = commit_directory + std::string("setup.bin");
            mcs_gen_setup(setup_path, config.vocab_size);

            auto write_quantized_commit = [&](float* ptr, size_t insize, size_t outsize, std::string fname) {
                size_t elements = insize * outsize;
                std::vector<int32_t> buffer(elements);
                for (size_t i = 0; i < elements; i++) {
                    buffer[i] = static_cast<int32_t>(ptr[i] * FP_W_SCALE);
                }
                std::cout << "Quantizing commit " << fname << " (" << elements << " elements)...\n";
                mcs_gen_commit(outsize, insize, buffer.data(), fname);
            };

            int64_t dim = config.dim;
            int64_t hidden_dim = config.hidden_dim;
            int64_t head_size = dim / config.n_heads;
            int64_t n_layers = config.n_layers;
            int64_t kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;

            for(int i = 0; i < config.n_layers; ++i) {
                std::cout << "Processing layer " << i << "...\n";
                float *wqkv = float_weights.wqkv + i * (dim + 2 * kv_dim) * dim;
                std::string fname = commit_directory + std::string("w8_") + std::to_string(i) + std::string(".bin");
                write_quantized_commit(wqkv, dim, (dim + 2 * kv_dim), fname);

                float *wo = float_weights.wo + i * dim * dim; // check dims
                fname = commit_directory + std::string("w3_") + std::to_string(i) + std::string(".bin");
                write_quantized_commit(wo, dim, dim, fname);

                float *w1w3 = float_weights.w1w3 + i * 2 * dim * hidden_dim;
                fname = commit_directory + std::string("w9_") + std::to_string(i) + std::string(".bin");
                write_quantized_commit(w1w3, dim, 2 * hidden_dim, fname);

                float *w2 = float_weights.w2 + i * hidden_dim * dim;
                fname = commit_directory + std::string("w5_") + std::to_string(i) + std::string(".bin");
                write_quantized_commit(w2, hidden_dim, dim, fname);
            }

            float *wcls = float_weights.wcls;
            std::string fname = commit_directory + std::string("w7.bin");
            write_quantized_commit(wcls, dim, config.vocab_size, fname);
        }
    }

    // Cleanup the mmap from the float file
    if (data != MAP_FAILED) { munmap(data, file_size); }
    if (fd != -1) { close(fd); }

    return 0;
}