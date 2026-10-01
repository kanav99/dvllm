#include "simdcrypt/SHA256.hpp"

#include <cstdio>
#include <cstring>
#include <string>

static bool check(const std::string& name, const char* data, size_t len, const char* expectHex)
{
    uint8_t hash[simdcrypt::SHA256::HashSize];
    simdcrypt::SHA256 hasher;
    hasher.Hash(reinterpret_cast<const uint8_t*>(data), len, hash);

    char hex[2 * simdcrypt::SHA256::HashSize + 1];
    for (size_t i = 0; i < simdcrypt::SHA256::HashSize; ++i) {
        snprintf(hex + 2 * i, 3, "%02x", hash[i]);
    }

    if (strcmp(hex, expectHex) != 0) {
        printf("[FAIL] %s\n  expected %s\n  got      %s\n", name.c_str(), expectHex, hex);
        return false;
    }
    printf("[ ok ] %s\n", name.c_str());
    return true;
}

int main()
{
    bool ok = true;

    // FIPS 180-4 / standard SHA-256 test vectors.
    ok &= check("empty", "", 0,
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    ok &= check("abc", "abc", 3,
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    ok &= check("two-block",
                "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 56,
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    // Streaming API in several chunks must equal the one-shot result.
    {
        const char* msg = "The quick brown fox jumps over the lazy dog";
        const char* expect = "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592";
        uint8_t hash[simdcrypt::SHA256::HashSize];
        simdcrypt::SHA256 hasher;
        hasher.Update(reinterpret_cast<const uint8_t*>(msg), 10);
        hasher.Update(reinterpret_cast<const uint8_t*>(msg + 10), 20);
        hasher.Update(reinterpret_cast<const uint8_t*>(msg + 30), strlen(msg) - 30);
        hasher.Final(hash);
        char hex[2 * simdcrypt::SHA256::HashSize + 1];
        for (size_t i = 0; i < simdcrypt::SHA256::HashSize; ++i)
            snprintf(hex + 2 * i, 3, "%02x", hash[i]);
        if (strcmp(hex, expect) != 0) {
            printf("[FAIL] streaming\n  expected %s\n  got      %s\n", expect, hex);
            ok = false;
        } else {
            printf("[ ok ] streaming\n");
        }
    }

    // A message spanning many blocks (1000 'a's).
    {
        std::string big(1000, 'a');
        ok &= check("1000-a", big.data(), big.size(),
                    "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
    }

    return ok ? 0 : 1;
}
