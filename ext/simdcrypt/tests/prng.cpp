#include "simdcrypt/PRNG.hpp"

using namespace simdcrypt;

int main() {
    block seed = toBlock(0x0123456789ABCDEFULL, 0x0FEDCBA987654321ULL);
    PRNG prng(seed);

    // just check if any errors are thrown
    // Generate some random numbers
    for (int i = 0; i < 10; ++i) {
        uint32_t rand_num = prng.get<uint32_t>();
        printf("Random number %d: %u\n", i, rand_num);
    }

    for (int i = 0; i < 10; ++i) {
        uint8_t rand_num = prng.get<uint8_t>();
        printf("Random number %d: %u\n", i, rand_num);
    }

    for (int i = 0; i < 10; ++i) {
        uint64_t rand_num = prng.get<uint64_t>();
        printf("Random number %d: %llu\n", i, rand_num);
    }

    for (int i = 0; i < 10; ++i) {
        block rand_num = prng.get<block>();
        uint64_t high = extract_u64<1>(rand_num);
        uint64_t low = extract_u64<0>(rand_num);
        printf("Random number %d: %016llx%016llx\n", i, high, low);
    }

    return 0;
}