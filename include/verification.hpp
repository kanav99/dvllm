#pragma once

#include <cstdint>
#include <iostream>
#include "simdcrypt/PRNG.hpp"
#include "constants.hpp"
#include <map>

extern "C" {
  extern void hello_rust();
  extern void rust_mcs_gen_setup(const char *fname, size_t max_dim);
  extern void rust_mcs_load_setup(const char *fname);
  extern void rust_mcs_gen_commit(uint d0, uint d1, int32_t *m, const char *fname);
  extern void rust_mcs_clear_commits();
  extern void rust_mcs_load_commit(int32_t layer, int32_t ID, const char *fname);
  extern uint rust_mcs_verify(int32_t layer, int32_t ID, uint d0, uint d1, __int128_t *x, __int128_t *y);
  extern uint rust_mcs_verify_batch(uint n, const int32_t *layers, const int32_t *ids,
                                     const uint *d0s, const uint *d1s,
                                     const __int128_t *x_concat, const __int128_t *y_concat);
}


void mcs_gen_setup(std::string fname, size_t max_dim) {
  rust_mcs_gen_setup(fname.c_str(), max_dim);
}
void mcs_load_setup(std::string fname) {
  rust_mcs_load_setup(fname.c_str());
}

void mcs_gen_commit(uint d0, uint d1, int32_t *m, std::string fname) {
  rust_mcs_gen_commit(d0, d1, m, fname.c_str());
}

void mcs_clear_commits() {
  rust_mcs_clear_commits();
}

void mcs_load_commit(int layer, int ID, std::string fname) {
  rust_mcs_load_commit(layer, ID, fname.c_str());
}

uint mcs_verify(int layer, int ID, uint d0, uint d1, __int128_t *x, __int128_t *y) {
  return rust_mcs_verify(layer, ID, d0, d1, x, y);
}

uint mcs_verify_batch(uint n, const int32_t *layers, const int32_t *ids,
                       const uint *d0s, const uint *d1s,
                       const __int128_t *x_concat, const __int128_t *y_concat) {
  return rust_mcs_verify_batch(n, layers, ids, d0s, d1s, x_concat, y_concat);
}

class Verifier {
  simdcrypt::PRNG prng;
  std::map<std::pair<int, int>, std::vector<__int128_t>> xs;
  std::map<std::pair<int, int>, std::vector<__int128_t>> ys;
  std::string prefix;

 public:
  Verifier(std::string prefix) : prefix(prefix), prng(simdcrypt::PRNG(simdcrypt::toBlock(rand(), rand()))) {}

  void Eat(int layer, int ID, integer_t *x, size_t len_x, integer_t *y, size_t len_y)
  {
    __int128_t coeff = __int128_t(prng.get<int64_t>());
    std::pair<int, int> id(layer, ID);

    if (xs.find(id) == xs.end()) {
      xs[id] = std::vector<__int128_t>(len_x, 0);
    } else {
      if (xs[id].size() != len_x) {
        throw std::invalid_argument("Inconsistent vector size");
      }
    }

    if (ys.find(id) == ys.end()) {
      ys[id] = std::vector<__int128_t>(len_y, 0);
    } else {
      if (ys[id].size() != len_y) {
        throw std::invalid_argument("Inconsistent vector size");
      }
    }

    for (size_t i = 0; i < len_x; ++i) {
      xs[id][i] += coeff * x[i];
    }

    for (size_t i = 0; i < len_y; ++i) {
      ys[id][i] += coeff * y[i];
    }
  }

  // Reads setup.bin and every commit file needed by the (layer, ID) pairs
  // seen so far, staging them in the Rust side's COMMITS table. Call this
  // before timing Verify() -- Verify() itself does no file I/O.
  void Load()
  {
    std::string setup_file = prefix + "/setup.bin";
    mcs_load_setup(setup_file.c_str());

    mcs_clear_commits();
    for (auto &pair : xs) {
      auto &id = pair.first;
      int layer = id.first;
      int ID = id.second;
      std::string s = prefix + "/w" + std::to_string(ID);
      if (ID != 7) {
        s = s + "_" + std::to_string(layer);
      }
      s = s + ".bin";
      mcs_load_commit(layer, ID, s);
    }
  }

  // Pure computation over data already staged by Load(). Performs no file
  // I/O. Instead of one MSM pair per (layer, ID), every id's contribution is
  // concatenated into a single pair of MSMs (one per side), which is
  // cheaper than N separate MSMs of the same total size. This is sound with
  // plain coefficient-1 concatenation because each xs[id]/ys[id] here is
  // already a sum of terms independently randomized by Eat()'s per-call
  // `coeff`, drawn fresh after that call's (x, y) is already fixed -- see
  // the comment on rust_mcs_verify_batch for the full argument.
  void Verify()
  {
    if (xs.empty()) return;

    std::vector<int32_t> layers, ids;
    std::vector<uint> d0s, d1s;
    std::vector<__int128_t> x_concat, y_concat;
    layers.reserve(xs.size());
    ids.reserve(xs.size());
    d0s.reserve(xs.size());
    d1s.reserve(xs.size());

    for (auto &pair : xs) {
      auto &id = pair.first;
      auto &x_vec = pair.second;
      auto &y_vec = ys.at(id);

      layers.push_back(id.first);
      ids.push_back(id.second);
      d1s.push_back((uint)x_vec.size());
      d0s.push_back((uint)y_vec.size());

      x_concat.insert(x_concat.end(), x_vec.begin(), x_vec.end());
      y_concat.insert(y_concat.end(), y_vec.begin(), y_vec.end());
    }

    bool ok = mcs_verify_batch((uint)layers.size(), layers.data(), ids.data(), d0s.data(), d1s.data(),
                                x_concat.data(), y_concat.data());
    if (ok) return;

    // The combined check only says "something's wrong", not which id --
    // fall back to per-id verification (the pre-batching path) to report
    // exactly which (layer, ID) pair is inconsistent.
    for (auto &pair : xs) {
      auto &id = pair.first;
      int layer = id.first;
      int ID = id.second;
      auto &x_vec = pair.second;
      auto &y_vec = ys.at(id);
      bool res = mcs_verify(layer, ID, y_vec.size(), x_vec.size(), x_vec.data(), y_vec.data());
      if (res == 0) {
        std::cout << "Verification failed for ID: " << ID << " layer: " << layer << std::endl;
      }
    }
  }
};
