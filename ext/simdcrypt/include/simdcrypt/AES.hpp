#pragma once

#include <iostream>
#include <limits>
#include <climits>
#include <cstring>
#include <cstdint>
#include <exception>
#include <iostream>

#if defined(USE_NEON_AES)
  #ifndef HARDWARE_ACCELERATION_ARM_NEON_AES
    #define HARDWARE_ACCELERATION_ARM_NEON_AES
  #endif
  #undef HARDWARE_ACCELERATION_INTEL_AESNI

#elif defined(USE_INTEL_AESNI)
  #ifndef HARDWARE_ACCELERATION_INTEL_AESNI
    #define HARDWARE_ACCELERATION_INTEL_AESNI
  #endif
  #undef HARDWARE_ACCELERATION_ARM_NEON_AES
#endif

#if (defined(__x86_64) || defined(__x86_64__) || defined(__amd64) || defined(__amd64__)) && !defined(USE_NEON_AES)
  #if defined(_MSC_VER)
    #include <intrin.h>
  #endif

  #include <emmintrin.h>
  #include <immintrin.h>
  #include <xmmintrin.h>

  #ifndef HARDWARE_ACCELERATION_INTEL_AESNI
    #define HARDWARE_ACCELERATION_INTEL_AESNI
  #endif
#elif (defined(__arm64__) || defined(__aarch64__) || defined(_M_ARM)) && !defined(USE_INTEL_AESNI)
  #if defined(__GNUC__)
    #include <stdint.h>
  #endif

  #if defined(__ARM_NEON) || defined(_MSC_VER)
    #include <arm_neon.h>
  #endif

  /* GCC and LLVM Clang, but not Apple Clang */
  #if defined(__GNUC__) && !defined(__apple_build_version__)
    #if defined(__ARM_ACLE) || defined(__ARM_FEATURE_CRYPTO)
      #include <arm_acle.h>
    #endif
  #endif

  #ifndef HARDWARE_ACCELERATION_ARM_NEON_AES
    #define HARDWARE_ACCELERATION_ARM_NEON_AES
  #endif
#else
  #error "AES hardware acceleration not supported on this platform"
#endif

namespace simdcrypt {

#ifdef HARDWARE_ACCELERATION_INTEL_AESNI
  typedef __m128i block;

  const block ZeroBlock = _mm_setzero_si128();

  inline block toBlock(const uint8_t* bytes) {
      return _mm_loadu_si128(reinterpret_cast<const __m128i*>(bytes));
  }

  inline block toBlock(uint64_t high, uint64_t low) {
      return _mm_set_epi64x(high, low);
  }

  inline block toBlock(uint64_t low) {
      return _mm_set_epi64x(0, low);
  }

  template <int i>
  uint64_t extract_u64(const block &b) {
      if constexpr (i == 0) {
          return static_cast<uint64_t>(_mm_cvtsi128_si64(b));
      } else if constexpr (i == 1) {
          return static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_unpackhi_epi64(b, b)));
      } else {
          static_assert(i < 2, "Index out of bounds for extract_u64<index>");
      }
  }

  template <int i>
  uint8_t extract_u8(const block &b) {
      return static_cast<uint8_t>(_mm_extract_epi8(b, i));
  }

  inline block load_block(const block* ptr) {
      return _mm_loadu_si128(ptr);
  }

  inline block xor_blocks(const block &a, const block &b) {
      return _mm_xor_si128(a, b);
  }

  inline void store_block(const block &b, uint8_t* dest) {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dest), b);
  }

#else
  typedef uint8x16_t block;

  const block ZeroBlock = vdupq_n_u8(0);

  inline block toBlock(const uint8_t* bytes) {
      return vld1q_u8(bytes);
  }

  inline block toBlock(uint64_t high, uint64_t low) {
      return vcombine_u8(vcreate_u8(low), vcreate_u8(high));
  }

  inline block toBlock(uint64_t low) {
      return vcombine_u8(vcreate_u8(low), vdup_n_u8(0));
  }

  inline block load_block(const block* ptr) {
      return vld1q_u8(reinterpret_cast<const uint8_t*>(ptr));
  }

  template <int i>
  uint64_t extract_u64(const block &b) {
      if constexpr (i == 0) {
          return vgetq_lane_u64(vreinterpretq_u64_u8(b), 0);
      } else if constexpr (i == 1) {
          return vgetq_lane_u64(vreinterpretq_u64_u8(b), 1);
      } else {
          static_assert(i < 2, "Index out of bounds for extract_u64<index>");
      }
  }

  template <int i>
  uint8_t extract_u8(const block &b) {
      return vgetq_lane_u8(b, i);
  }

  inline block xor_blocks(const block &a, const block &b) {
      return veorq_u8(a, b);
  }

  inline void store_block(const block &b, uint8_t* dest) {
      vst1q_u8(dest, b);
  }

#endif

  class AES {
    public:
      AES(block key = ZeroBlock);
      block get_round_key(int round) const {
          return round_keys[round];
      }
      void set_key(const block& key) {
          *this = AES(key);
      }

      void ecbEncBlock(const block &plaintext, block &ciphertext) const;
      block ecbEncBlock(const block &plaintext) const;
      void ecbEncCounterMode(uint64_t baseIdx, uint64_t blockLength, block *ciphertext) const;

    private:
      block round_keys[11];
  };
} // namespace simdcrypt
