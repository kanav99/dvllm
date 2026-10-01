#include "simdcrypt/AES.hpp"
#include <utility>

using namespace simdcrypt;

template <size_t... Ints, typename F>
constexpr void constexpr_for_impl(std::index_sequence<Ints...>, F&& function) {
    // This uses a fold expression to apply the function to each index
    (std::forward<F>(function)(std::integral_constant<size_t, Ints>{}), ...);
}

template <size_t Size, typename F>
constexpr void constexpr_for(F&& function) {
    constexpr_for_impl(std::make_index_sequence<Size>(), std::forward<F>(function));
}

int main() {

    uint64_t high = 0x1122334455667788ULL;
    uint64_t low  = 0x99AABBCCDDEEFF00ULL;
    block b = toBlock(high, low);
    if (extract_u64<1>(b) != high || extract_u64<0>(b) != low) {
        return 1;
    }

    constexpr_for<16>([&](auto i) {
        uint8_t expected = ((uint8_t*)&b)[i];
        if (extract_u8<i>(b) != expected) {
            throw std::runtime_error("extract_u8 failed");
        }
    });


    high = time(NULL);
    high = (high << 32) | time(NULL);
    low = time(NULL);
    low = (low << 32) | time(NULL);
    b = toBlock(high, low);
    if (extract_u64<1>(b) != high || extract_u64<0>(b) != low) {
        return 1;
    }

    constexpr_for<16>([&](auto i) {
        uint8_t expected = ((uint8_t*)&b)[i];
        if (extract_u8<i>(b) != expected) {
            throw std::runtime_error("extract_u8 failed");
        }
    });

    return 0;
}