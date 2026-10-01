#include "simdcrypt/AES.hpp"
#include "openssl/aes.h"
#include <utility>

using namespace simdcrypt;

block random_block() {
    uint8_t bytes[16];
    for (int i = 0; i < 16; ++i) {
        bytes[i] = rand() % 256;
    }
    return toBlock(bytes);
}

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

std::string BlockToString(const block& v) {
    std::string s;
    constexpr_for<16>([&](auto i) {
        s.push_back(static_cast<char>(extract_u8<i>(v)));
    });
    return s;
}

int main() {

    block key = random_block();
    block plaintext = random_block();

    AES aes(key);
    block ciphertext = aes.ecbEncBlock(plaintext);

    // OpenSSL based implementation.
    std::string s_key = BlockToString(key);
    std::string s_plaintext = BlockToString(plaintext);

    AES_KEY aes_key;
    if (AES_set_encrypt_key(reinterpret_cast<const uint8_t*>(s_key.data()),
                            128, &aes_key) != 0) {
        return 1;
    }

    std::string s_ciphertext(16, '\0');
    AES_ecb_encrypt(reinterpret_cast<const uint8_t*>(s_plaintext.data()),
                    reinterpret_cast<uint8_t*>(&s_ciphertext[0]),
                    &aes_key, AES_ENCRYPT);
    std::string expected_ciphertext = BlockToString(ciphertext);

    if (s_ciphertext != expected_ciphertext) {
        return 1;
    }

}
