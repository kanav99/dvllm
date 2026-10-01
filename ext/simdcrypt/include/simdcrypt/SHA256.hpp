#pragma once

// SHA-256 (FIPS 180-4).
//
// This header exposes a small streaming SHA-256 implementation with the same
// surface as AESHash (Update / Final / Hash / HashSize) so it can be used as a
// drop-in replacement.
//
// The block compression function is dispatched at compile time:
//   * x86-64  with the SHA extension (-msha / -march=native on a capable CPU)
//             -> Intel SHA-NI intrinsics (_mm_sha256*).
//   * AArch64 with the crypto/SHA2 extension (-march=armv8-a+crypto)
//             -> ARMv8 SHA2 intrinsics (vsha256*).
//   * everything else -> a portable scalar fallback.
//
// The scalar fallback is always compiled, so the class produces correct results
// on every machine regardless of available hardware acceleration.

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <iostream>

// --- intrinsics detection --------------------------------------------------
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64) || \
    defined(__i386__)   || defined(_M_IX86)
  #if defined(_MSC_VER)
    #include <intrin.h>
  #endif
  #include <immintrin.h>
  #if defined(__SHA__)
    #define SIMDCRYPT_SHA256_X86 1
  #endif
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || defined(_M_ARM)
  #if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
    #include <arm_neon.h>
  #endif
  #if defined(__ARM_FEATURE_SHA2) || defined(__ARM_FEATURE_CRYPTO)
    #define SIMDCRYPT_SHA256_ARM 1
  #endif
#endif

namespace simdcrypt {
namespace sha256_detail {

    // SHA-256 round constants (FIPS 180-4, section 4.2.2).
    alignas(16) static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    inline uint32_t ror(uint32_t x, int n) {
        return (x >> n) | (x << (32 - n));
    }

    // Portable scalar compression over one or more 64-byte blocks.
    inline void compress_scalar(uint32_t state[8], const uint8_t* data, size_t length) {
        while (length >= 64) {
            uint32_t w[64];
            for (int i = 0; i < 16; ++i) {
                w[i] = (uint32_t(data[i * 4 + 0]) << 24) |
                       (uint32_t(data[i * 4 + 1]) << 16) |
                       (uint32_t(data[i * 4 + 2]) <<  8) |
                       (uint32_t(data[i * 4 + 3]));
            }
            for (int i = 16; i < 64; ++i) {
                uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
                uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }

            uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
            uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

            for (int i = 0; i < 64; ++i) {
                uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
                uint32_t ch = (e & f) ^ (~e & g);
                uint32_t t1 = h + S1 + ch + K[i] + w[i];
                uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
                uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                uint32_t t2 = S0 + maj;
                h = g; g = f; f = e; e = d + t1;
                d = c; c = b; b = a; a = t1 + t2;
            }

            state[0] += a; state[1] += b; state[2] += c; state[3] += d;
            state[4] += e; state[5] += f; state[6] += g; state[7] += h;

            data += 64;
            length -= 64;
        }
    }

#if defined(SIMDCRYPT_SHA256_X86)
    // Intel SHA-NI. Adapted from Jeffrey Walton's public-domain reference
    // (https://github.com/noloader/SHA-Intrinsics).
    inline void compress_x86(uint32_t state[8], const uint8_t* data, size_t length) {
        __m128i STATE0, STATE1;
        __m128i MSG, TMP;
        __m128i MSG0, MSG1, MSG2, MSG3;
        __m128i ABEF_SAVE, CDGH_SAVE;
        const __m128i MASK = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);

        TMP    = _mm_loadu_si128((const __m128i*) &state[0]);
        STATE1 = _mm_loadu_si128((const __m128i*) &state[4]);

        TMP    = _mm_shuffle_epi32(TMP, 0xB1);          // CDAB
        STATE1 = _mm_shuffle_epi32(STATE1, 0x1B);       // EFGH
        STATE0 = _mm_alignr_epi8(TMP, STATE1, 8);       // ABEF
        STATE1 = _mm_blend_epi16(STATE1, TMP, 0xF0);    // CDGH

        while (length >= 64) {
            ABEF_SAVE = STATE0;
            CDGH_SAVE = STATE1;

            // Rounds 0-3
            MSG  = _mm_loadu_si128((const __m128i*) (data + 0));
            MSG0 = _mm_shuffle_epi8(MSG, MASK);
            MSG  = _mm_add_epi32(MSG0, _mm_set_epi64x(0xE9B5DBA5B5C0FBCFULL, 0x71374491428A2F98ULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

            // Rounds 4-7
            MSG1 = _mm_loadu_si128((const __m128i*) (data + 16));
            MSG1 = _mm_shuffle_epi8(MSG1, MASK);
            MSG  = _mm_add_epi32(MSG1, _mm_set_epi64x(0xAB1C5ED5923F82A4ULL, 0x59F111F13956C25BULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
            MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);

            // Rounds 8-11
            MSG2 = _mm_loadu_si128((const __m128i*) (data + 32));
            MSG2 = _mm_shuffle_epi8(MSG2, MASK);
            MSG  = _mm_add_epi32(MSG2, _mm_set_epi64x(0x550C7DC3243185BEULL, 0x12835B01D807AA98ULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
            MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);

            // Rounds 12-15
            MSG3 = _mm_loadu_si128((const __m128i*) (data + 48));
            MSG3 = _mm_shuffle_epi8(MSG3, MASK);
            MSG  = _mm_add_epi32(MSG3, _mm_set_epi64x(0xC19BF1749BDC06A7ULL, 0x80DEB1FE72BE5D74ULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            TMP  = _mm_alignr_epi8(MSG3, MSG2, 4);
            MSG0 = _mm_add_epi32(MSG0, TMP);
            MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
            MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);

            // Rounds 16-19
            MSG  = _mm_add_epi32(MSG0, _mm_set_epi64x(0x240CA1CC0FC19DC6ULL, 0xEFBE4786E49B69C1ULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            TMP  = _mm_alignr_epi8(MSG0, MSG3, 4);
            MSG1 = _mm_add_epi32(MSG1, TMP);
            MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
            MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);

            // Rounds 20-23
            MSG  = _mm_add_epi32(MSG1, _mm_set_epi64x(0x76F988DA5CB0A9DCULL, 0x4A7484AA2DE92C6FULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            TMP  = _mm_alignr_epi8(MSG1, MSG0, 4);
            MSG2 = _mm_add_epi32(MSG2, TMP);
            MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
            MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);

            // Rounds 24-27
            MSG  = _mm_add_epi32(MSG2, _mm_set_epi64x(0xBF597FC7B00327C8ULL, 0xA831C66D983E5152ULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            TMP  = _mm_alignr_epi8(MSG2, MSG1, 4);
            MSG3 = _mm_add_epi32(MSG3, TMP);
            MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
            MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);

            // Rounds 28-31
            MSG  = _mm_add_epi32(MSG3, _mm_set_epi64x(0x1429296706CA6351ULL, 0xD5A79147C6E00BF3ULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            TMP  = _mm_alignr_epi8(MSG3, MSG2, 4);
            MSG0 = _mm_add_epi32(MSG0, TMP);
            MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
            MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);

            // Rounds 32-35
            MSG  = _mm_add_epi32(MSG0, _mm_set_epi64x(0x53380D134D2C6DFCULL, 0x2E1B213827B70A85ULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            TMP  = _mm_alignr_epi8(MSG0, MSG3, 4);
            MSG1 = _mm_add_epi32(MSG1, TMP);
            MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
            MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);

            // Rounds 36-39
            MSG  = _mm_add_epi32(MSG1, _mm_set_epi64x(0x92722C8581C2C92EULL, 0x766A0ABB650A7354ULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            TMP  = _mm_alignr_epi8(MSG1, MSG0, 4);
            MSG2 = _mm_add_epi32(MSG2, TMP);
            MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
            MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);

            // Rounds 40-43
            MSG  = _mm_add_epi32(MSG2, _mm_set_epi64x(0xC76C51A3C24B8B70ULL, 0xA81A664BA2BFE8A1ULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            TMP  = _mm_alignr_epi8(MSG2, MSG1, 4);
            MSG3 = _mm_add_epi32(MSG3, TMP);
            MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
            MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);

            // Rounds 44-47
            MSG  = _mm_add_epi32(MSG3, _mm_set_epi64x(0x106AA070F40E3585ULL, 0xD6990624D192E819ULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            TMP  = _mm_alignr_epi8(MSG3, MSG2, 4);
            MSG0 = _mm_add_epi32(MSG0, TMP);
            MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
            MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);

            // Rounds 48-51
            MSG  = _mm_add_epi32(MSG0, _mm_set_epi64x(0x34B0BCB52748774CULL, 0x1E376C0819A4C116ULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            TMP  = _mm_alignr_epi8(MSG0, MSG3, 4);
            MSG1 = _mm_add_epi32(MSG1, TMP);
            MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
            MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);

            // Rounds 52-55
            MSG  = _mm_add_epi32(MSG1, _mm_set_epi64x(0x682E6FF35B9CCA4FULL, 0x4ED8AA4A391C0CB3ULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            TMP  = _mm_alignr_epi8(MSG1, MSG0, 4);
            MSG2 = _mm_add_epi32(MSG2, TMP);
            MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

            // Rounds 56-59
            MSG  = _mm_add_epi32(MSG2, _mm_set_epi64x(0x8CC7020884C87814ULL, 0x78A5636F748F82EEULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            TMP  = _mm_alignr_epi8(MSG2, MSG1, 4);
            MSG3 = _mm_add_epi32(MSG3, TMP);
            MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

            // Rounds 60-63
            MSG  = _mm_add_epi32(MSG3, _mm_set_epi64x(0xC67178F2BEF9A3F7ULL, 0xA4506CEB90BEFFFAULL));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            MSG  = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

            STATE0 = _mm_add_epi32(STATE0, ABEF_SAVE);
            STATE1 = _mm_add_epi32(STATE1, CDGH_SAVE);

            data += 64;
            length -= 64;
        }

        TMP    = _mm_shuffle_epi32(STATE0, 0x1B);       // FEBA
        STATE1 = _mm_shuffle_epi32(STATE1, 0xB1);       // DCHG
        STATE0 = _mm_blend_epi16(TMP, STATE1, 0xF0);    // DCBA
        STATE1 = _mm_alignr_epi8(STATE1, TMP, 8);       // ABEF

        _mm_storeu_si128((__m128i*) &state[0], STATE0);
        _mm_storeu_si128((__m128i*) &state[4], STATE1);
    }
#endif // SIMDCRYPT_SHA256_X86

#if defined(SIMDCRYPT_SHA256_ARM)
    // ARMv8 SHA2. Adapted from Jeffrey Walton's public-domain reference
    // (https://github.com/noloader/SHA-Intrinsics).
    inline void compress_arm(uint32_t state[8], const uint8_t* data, size_t length) {
        uint32x4_t STATE0, STATE1, ABEF_SAVE, CDGH_SAVE;
        uint32x4_t MSG0, MSG1, MSG2, MSG3;
        uint32x4_t TMP0, TMP1, TMP2;

        STATE0 = vld1q_u32(&state[0]);
        STATE1 = vld1q_u32(&state[4]);

        while (length >= 64) {
            ABEF_SAVE = STATE0;
            CDGH_SAVE = STATE1;

            MSG0 = vld1q_u32((const uint32_t*)(data +  0));
            MSG1 = vld1q_u32((const uint32_t*)(data + 16));
            MSG2 = vld1q_u32((const uint32_t*)(data + 32));
            MSG3 = vld1q_u32((const uint32_t*)(data + 48));

            MSG0 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(MSG0)));
            MSG1 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(MSG1)));
            MSG2 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(MSG2)));
            MSG3 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(MSG3)));

            TMP0 = vaddq_u32(MSG0, vld1q_u32(&K[0x00]));

            // Rounds 0-3
            MSG0 = vsha256su0q_u32(MSG0, MSG1);
            TMP2 = STATE0;
            TMP1 = vaddq_u32(MSG1, vld1q_u32(&K[0x04]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
            MSG0 = vsha256su1q_u32(MSG0, MSG2, MSG3);

            // Rounds 4-7
            MSG1 = vsha256su0q_u32(MSG1, MSG2);
            TMP2 = STATE0;
            TMP0 = vaddq_u32(MSG2, vld1q_u32(&K[0x08]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
            MSG1 = vsha256su1q_u32(MSG1, MSG3, MSG0);

            // Rounds 8-11
            MSG2 = vsha256su0q_u32(MSG2, MSG3);
            TMP2 = STATE0;
            TMP1 = vaddq_u32(MSG3, vld1q_u32(&K[0x0c]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
            MSG2 = vsha256su1q_u32(MSG2, MSG0, MSG1);

            // Rounds 12-15
            MSG3 = vsha256su0q_u32(MSG3, MSG0);
            TMP2 = STATE0;
            TMP0 = vaddq_u32(MSG0, vld1q_u32(&K[0x10]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
            MSG3 = vsha256su1q_u32(MSG3, MSG1, MSG2);

            // Rounds 16-19
            MSG0 = vsha256su0q_u32(MSG0, MSG1);
            TMP2 = STATE0;
            TMP1 = vaddq_u32(MSG1, vld1q_u32(&K[0x14]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
            MSG0 = vsha256su1q_u32(MSG0, MSG2, MSG3);

            // Rounds 20-23
            MSG1 = vsha256su0q_u32(MSG1, MSG2);
            TMP2 = STATE0;
            TMP0 = vaddq_u32(MSG2, vld1q_u32(&K[0x18]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
            MSG1 = vsha256su1q_u32(MSG1, MSG3, MSG0);

            // Rounds 24-27
            MSG2 = vsha256su0q_u32(MSG2, MSG3);
            TMP2 = STATE0;
            TMP1 = vaddq_u32(MSG3, vld1q_u32(&K[0x1c]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
            MSG2 = vsha256su1q_u32(MSG2, MSG0, MSG1);

            // Rounds 28-31
            MSG3 = vsha256su0q_u32(MSG3, MSG0);
            TMP2 = STATE0;
            TMP0 = vaddq_u32(MSG0, vld1q_u32(&K[0x20]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
            MSG3 = vsha256su1q_u32(MSG3, MSG1, MSG2);

            // Rounds 32-35
            MSG0 = vsha256su0q_u32(MSG0, MSG1);
            TMP2 = STATE0;
            TMP1 = vaddq_u32(MSG1, vld1q_u32(&K[0x24]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
            MSG0 = vsha256su1q_u32(MSG0, MSG2, MSG3);

            // Rounds 36-39
            MSG1 = vsha256su0q_u32(MSG1, MSG2);
            TMP2 = STATE0;
            TMP0 = vaddq_u32(MSG2, vld1q_u32(&K[0x28]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
            MSG1 = vsha256su1q_u32(MSG1, MSG3, MSG0);

            // Rounds 40-43
            MSG2 = vsha256su0q_u32(MSG2, MSG3);
            TMP2 = STATE0;
            TMP1 = vaddq_u32(MSG3, vld1q_u32(&K[0x2c]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
            MSG2 = vsha256su1q_u32(MSG2, MSG0, MSG1);

            // Rounds 44-47
            MSG3 = vsha256su0q_u32(MSG3, MSG0);
            TMP2 = STATE0;
            TMP0 = vaddq_u32(MSG0, vld1q_u32(&K[0x30]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
            MSG3 = vsha256su1q_u32(MSG3, MSG1, MSG2);

            // Rounds 48-51
            TMP2 = STATE0;
            TMP1 = vaddq_u32(MSG1, vld1q_u32(&K[0x34]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);

            // Rounds 52-55
            TMP2 = STATE0;
            TMP0 = vaddq_u32(MSG2, vld1q_u32(&K[0x38]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);

            // Rounds 56-59
            TMP2 = STATE0;
            TMP1 = vaddq_u32(MSG3, vld1q_u32(&K[0x3c]));
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);

            // Rounds 60-63
            TMP2 = STATE0;
            STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
            STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);

            STATE0 = vaddq_u32(STATE0, ABEF_SAVE);
            STATE1 = vaddq_u32(STATE1, CDGH_SAVE);

            data += 64;
            length -= 64;
        }

        vst1q_u32(&state[0], STATE0);
        vst1q_u32(&state[4], STATE1);
    }
#endif // SIMDCRYPT_SHA256_ARM

    // Compress `length` bytes (a whole number of 64-byte blocks) into `state`.
    inline void compress(uint32_t state[8], const uint8_t* data, size_t length) {
#if defined(SIMDCRYPT_SHA256_X86)
        compress_x86(state, data, length);
#elif defined(SIMDCRYPT_SHA256_ARM)
        compress_arm(state, data, length);
#else
        compress_scalar(state, data, length);
#endif
    }

} // namespace sha256_detail

    // Streaming SHA-256. Interface mirrors AESHash so the two are interchangeable.
    class SHA256 {
        uint32_t mState[8];
        uint8_t  mBuffer[64];
        size_t   mBufLen;
        uint64_t mTotalLen;

    public:
        static constexpr size_t HashSize  = 32;
        static constexpr size_t BlockSize = 64;

        SHA256() { reset(); }

        void reset() {
            mState[0] = 0x6a09e667; mState[1] = 0xbb67ae85;
            mState[2] = 0x3c6ef372; mState[3] = 0xa54ff53a;
            mState[4] = 0x510e527f; mState[5] = 0x9b05688c;
            mState[6] = 0x1f83d9ab; mState[7] = 0x5be0cd19;
            mBufLen = 0;
            mTotalLen = 0;
        }

        void Update(const uint8_t* data, size_t length) {
            mTotalLen += length;

            // Top up a partially filled block first.
            if (mBufLen) {
                size_t need = BlockSize - mBufLen;
                size_t take = length < need ? length : need;
                memcpy(mBuffer + mBufLen, data, take);
                mBufLen += take;
                data += take;
                length -= take;
                if (mBufLen == BlockSize) {
                    sha256_detail::compress(mState, mBuffer, BlockSize);
                    mBufLen = 0;
                }
            }

            // Bulk process whole blocks straight from the input.
            if (length >= BlockSize) {
                size_t nblocks = length / BlockSize;
                sha256_detail::compress(mState, data, nblocks * BlockSize);
                data += nblocks * BlockSize;
                length -= nblocks * BlockSize;
            }

            // Stash the remainder.
            if (length) {
                memcpy(mBuffer, data, length);
                mBufLen = length;
            }
        }

        void Final(uint8_t* hash) {
            uint64_t bitLen = mTotalLen * 8;

            // Append the 0x80 terminator, then zero-pad and append the 64-bit
            // big-endian message length so the total is a whole block.
            mBuffer[mBufLen++] = 0x80;
            if (mBufLen > 56) {
                while (mBufLen < BlockSize) mBuffer[mBufLen++] = 0;
                sha256_detail::compress(mState, mBuffer, BlockSize);
                mBufLen = 0;
            }
            while (mBufLen < 56) mBuffer[mBufLen++] = 0;
            for (int i = 7; i >= 0; --i) {
                mBuffer[mBufLen++] = static_cast<uint8_t>(bitLen >> (i * 8));
            }
            sha256_detail::compress(mState, mBuffer, BlockSize);

            // Serialize the state, big-endian, into the digest.
            for (int i = 0; i < 8; ++i) {
                hash[i * 4 + 0] = static_cast<uint8_t>(mState[i] >> 24);
                hash[i * 4 + 1] = static_cast<uint8_t>(mState[i] >> 16);
                hash[i * 4 + 2] = static_cast<uint8_t>(mState[i] >>  8);
                hash[i * 4 + 3] = static_cast<uint8_t>(mState[i]);
            }

            reset();
        }

        // One-shot convenience: hash `length` bytes into `hash` (32 bytes).
        void Hash(const uint8_t* data, size_t length, uint8_t* hash) {
            reset();
            Update(data, length);
            Final(hash);
        }
    };

} // namespace simdcrypt
