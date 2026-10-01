#include "simdcrypt/AESHash.hpp"

int main()
{
    simdcrypt::AESHash hasher;
    const char* data = "Hello, world!";
    hasher.Update(reinterpret_cast<const uint8_t*>(data), strlen(data));

    uint8_t hash[simdcrypt::AESHash::HashSize];
    hasher.Final(hash);

    uint8_t expectedHash[simdcrypt::AESHash::HashSize] = {
        0x80, 0x5d, 0x0b, 0x40, 0xf7, 0x93, 0xa0, 0x87, 0x19, 0x31, 0x02, 0x09, 0xdd, 0x32, 0xc3, 0x5d
    };
    // Print the hash
    for (size_t i = 0; i < simdcrypt::AESHash::HashSize; ++i) {
        if (hash[i] != expectedHash[i]) {
            printf("Hash mismatch at byte %zu: expected %02x, got %02x\n", i, expectedHash[i], hash[i]);
            return 1;
        }
    }

    return 0;
}