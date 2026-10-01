#pragma once

#include <list>
#include <unordered_map>
#include <vector>


const int MAX_CACHE_SIZE = 0; // Adjust this capacity as needed
// TODO: disabling cache due to PRNG optimization and fair eval

class LRUEmbeddingCache {
private:
    std::list<int> lru_list;
    // Maps token ID to a pair: {embedding_vector, iterator_to_list_node}
    std::unordered_map<int, std::pair<std::vector<float>, std::list<int>::iterator>> cache_map;

public:
    bool get(int token, float* out_embedding, int dim) {
        if (MAX_CACHE_SIZE == 0) {
            return false; // Cache is disabled
        }
        auto it = cache_map.find(token);
        if (it == cache_map.end()) {
            return false; // Cache miss
        }
        
        // Cache hit: Move this token to the front of the LRU access list
        lru_list.splice(lru_list.begin(), lru_list, it->second.second);
        
        // Copy the cached embedding to the output pointer
        memcpy(out_embedding, it->second.first.data(), dim * sizeof(float));
        return true;
    }

    void put(int token, const float* in_embedding, int dim) {
        if (MAX_CACHE_SIZE == 0) {
            return; // Cache is disabled
        }
        // If it's already in the cache, skip adding it
        if (cache_map.find(token) != cache_map.end()) {
            return;
        }

        // If cache is at maximum capacity, evict the least recently used item (back of the list)
        if (cache_map.size() >= MAX_CACHE_SIZE) {
            int lru_token = lru_list.back();
            lru_list.pop_back();
            cache_map.erase(lru_token);
        }

        // Add the new token to the front (most recently used)
        lru_list.push_front(token);
        
        // Store the cloned embedding and its list location in the map
        std::vector<float> emb(in_embedding, in_embedding + dim);
        cache_map[token] = {std::move(emb), lru_list.begin()};
    }
};
