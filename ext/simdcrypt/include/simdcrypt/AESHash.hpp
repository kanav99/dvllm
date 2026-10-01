#pragma once

#include "AES.hpp"
#include <vector>

namespace simdcrypt
{
    // AES based hash from https://csrc.nist.rip/groups/ST/toolkit/BCM/documents/proposedmodes/aes-hash/aeshash.pdf
    class AESHash
    {
        std::vector<uint8_t> mBuffer;
    public:
        static constexpr size_t HashSize = 16;
        AESHash(): mBuffer()
        {
        }

        void Update(const uint8_t* data, size_t length)
        {
            mBuffer.insert(mBuffer.end(), data, data + length);
        }

        void Final(uint8_t* hash)
        {
            for (size_t i = mBuffer.size(); i < ((mBuffer.size() + 15) / 16) * 16; ++i)
            {
                mBuffer.push_back(0);
            }

            block h = toBlock(-1ull, -1ull);
            for (size_t i = 0; i < mBuffer.size(); i += 16)
            {
                block b = load_block((block *)(&mBuffer[i]));
                AES aes(b);
                block e = aes.ecbEncBlock(h);
                h = xor_blocks(h, e);
            }
            store_block(h, hash);

            mBuffer.clear();
        }

        void Hash(const uint8_t* data, size_t length, uint8_t* hash)
        {
            if (length % 16 != 0)
            {
                std::cerr << "Error: Input length must be a multiple of 16" << std::endl;
                return;
            }

            block h = toBlock(-1ull, -1ull);
            for (size_t i = 0; i < length; i += 16)
            {
                block b = load_block((block *)(&data[i]));
                AES aes(b);
                block e = aes.ecbEncBlock(h);
                h = xor_blocks(h, e);
            }
            store_block(h, hash);
        }
    };
} // namespace simdcrypt
