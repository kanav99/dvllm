#include "simdcrypt/AES.hpp"
#include <cstdint>

namespace simdcrypt {

#ifdef HARDWARE_ACCELERATION_INTEL_AESNI

template <int rcon>
block aes_128_key_expansion(block key){
    block keygened = _mm_aeskeygenassist_si128(key, rcon);
	keygened = _mm_shuffle_epi32(keygened, _MM_SHUFFLE(3,3,3,3));
	key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
	key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
	key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
	return _mm_xor_si128(key, keygened);
}

#else

template <int rcon>
block aes_128_key_expansion(block key){
    uint8x16_t temp = vaeseq_u8(key, vdupq_n_u8(0x00));
    // printblock(temp);
    uint32_t t = (vgetq_lane_u8(temp, 9) ^ rcon)  |
                 (vgetq_lane_u8(temp, 6) << 8)  |
                 (vgetq_lane_u8(temp, 3) << 16) |
                 (vgetq_lane_u8(temp, 12) << 24);
    uint8x16_t keygened = vdupq_n_u32(t);
    key = veorq_u8(key, vextq_u8(vdupq_n_u8(0), key, 12));
    key = veorq_u8(key, vextq_u8(vdupq_n_u8(0), key, 12));
    key = veorq_u8(key, vextq_u8(vdupq_n_u8(0), key, 12));
    return veorq_u8(key, keygened);
}

#endif

AES::AES(block key) {
    round_keys[0]  = key;
    round_keys[1]  = aes_128_key_expansion<0x01>(round_keys[0]);
    round_keys[2]  = aes_128_key_expansion<0x02>(round_keys[1]);
    round_keys[3]  = aes_128_key_expansion<0x04>(round_keys[2]);
    round_keys[4]  = aes_128_key_expansion<0x08>(round_keys[3]);
    round_keys[5]  = aes_128_key_expansion<0x10>(round_keys[4]);
    round_keys[6]  = aes_128_key_expansion<0x20>(round_keys[5]);
    round_keys[7]  = aes_128_key_expansion<0x40>(round_keys[6]);
    round_keys[8]  = aes_128_key_expansion<0x80>(round_keys[7]);
    round_keys[9]  = aes_128_key_expansion<0x1B>(round_keys[8]);
    round_keys[10] = aes_128_key_expansion<0x36>(round_keys[9]);
}

    void AES::ecbEncBlock(const block & plaintext, block &ciphertext) const
    {
#if defined(HARDWARE_ACCELERATION_INTEL_AESNI)
        ciphertext = _mm_xor_si128(plaintext, round_keys[0]);
        ciphertext = _mm_aesenc_si128(ciphertext, round_keys[1]);
        ciphertext = _mm_aesenc_si128(ciphertext, round_keys[2]);
        ciphertext = _mm_aesenc_si128(ciphertext, round_keys[3]);
        ciphertext = _mm_aesenc_si128(ciphertext, round_keys[4]);
        ciphertext = _mm_aesenc_si128(ciphertext, round_keys[5]);
        ciphertext = _mm_aesenc_si128(ciphertext, round_keys[6]);
        ciphertext = _mm_aesenc_si128(ciphertext, round_keys[7]);
        ciphertext = _mm_aesenc_si128(ciphertext, round_keys[8]);
        ciphertext = _mm_aesenc_si128(ciphertext, round_keys[9]);
        ciphertext = _mm_aesenclast_si128(ciphertext, round_keys[10]);
#elif defined(HARDWARE_ACCELERATION_ARM_NEON_AES)
        ciphertext = vaesmcq_u8(vaeseq_u8(plaintext, round_keys[0]));
        ciphertext = vaesmcq_u8(vaeseq_u8(ciphertext, round_keys[1]));
        ciphertext = vaesmcq_u8(vaeseq_u8(ciphertext, round_keys[2]));
        ciphertext = vaesmcq_u8(vaeseq_u8(ciphertext, round_keys[3]));
        ciphertext = vaesmcq_u8(vaeseq_u8(ciphertext, round_keys[4]));
        ciphertext = vaesmcq_u8(vaeseq_u8(ciphertext, round_keys[5]));
        ciphertext = vaesmcq_u8(vaeseq_u8(ciphertext, round_keys[6]));
        ciphertext = vaesmcq_u8(vaeseq_u8(ciphertext, round_keys[7]));
        ciphertext = vaesmcq_u8(vaeseq_u8(ciphertext, round_keys[8]));
        ciphertext = vaeseq_u8(ciphertext, round_keys[9]);
        ciphertext = veorq_u8(ciphertext, round_keys[10]);
#endif
    }

    block AES::ecbEncBlock(const block & plaintext) const
    {
        block ciphertext;
        ecbEncBlock(plaintext, ciphertext);
        return ciphertext;
    }

    void AES::ecbEncCounterMode(uint64_t baseIdx, uint64_t blockLength, block *ciphertext) const
    {
        for (uint64_t i = 0; i < blockLength; ++i)
        {
            uint64_t counter = baseIdx + i;

#if defined(HARDWARE_ACCELERATION_INTEL_AESNI)
            block input = _mm_set1_epi64x(counter);
#elif defined(HARDWARE_ACCELERATION_ARM_NEON_AES)
            block input = vdupq_n_u64(counter);
#endif
            ciphertext[i] = ecbEncBlock(input);
        }
    }

} // namespace simdcrypt