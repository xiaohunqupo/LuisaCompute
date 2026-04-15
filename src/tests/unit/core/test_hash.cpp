// Test for hash utilities.
// Covers: hash64, hash128/Hash128, hash_value, hash_combine,
//         string_hash, basic_string_hash, determinism checks.

#include "ut/ut.hpp"

#include <cstring>
#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/logging.h>

using namespace boost::ut;
using namespace boost::ut::literals;

// ---- hash64 ----

static inline const auto reg_hash64_basic = [] {
    "hash64_basic"_test = [] {
        int value = 42;
        auto h = luisa::hash64(&value, sizeof(value), luisa::hash64_default_seed);
        // hash should be non-zero for non-trivial input
        expect(h != 0u) << "hash64 of a non-zero value should not be zero";
    };
    return 0;
}();

static inline const auto reg_hash64_deterministic = [] {
    "hash64_deterministic"_test = [] {
        const char data[] = "hello world";
        auto h1 = luisa::hash64(data, sizeof(data), luisa::hash64_default_seed);
        auto h2 = luisa::hash64(data, sizeof(data), luisa::hash64_default_seed);
        expect(h1 == h2) << "same data and seed must produce same hash";
    };
    return 0;
}();

static inline const auto reg_hash64_different_data = [] {
    "hash64_different_data"_test = [] {
        int a = 1, b = 2;
        auto ha = luisa::hash64(&a, sizeof(a), luisa::hash64_default_seed);
        auto hb = luisa::hash64(&b, sizeof(b), luisa::hash64_default_seed);
        expect(ha != hb) << "different data should (almost certainly) produce different hashes";
    };
    return 0;
}();

static inline const auto reg_hash64_different_seeds = [] {
    "hash64_different_seeds"_test = [] {
        int value = 42;
        auto h1 = luisa::hash64(&value, sizeof(value), 0u);
        auto h2 = luisa::hash64(&value, sizeof(value), 1u);
        expect(h1 != h2) << "different seeds should (almost certainly) produce different hashes";
    };
    return 0;
}();

static inline const auto reg_hash64_empty = [] {
    "hash64_empty_data"_test = [] {
        // Hashing zero-length data should still work
        auto h = luisa::hash64(nullptr, 0u, luisa::hash64_default_seed);
        // just verify it doesn't crash; result is implementation-defined
        (void)h;
        expect(true);
    };
    return 0;
}();

// ---- hash<T> for arithmetic types ----

static inline const auto reg_hash_arithmetic = [] {
    "hash_arithmetic"_test = [] {
        luisa::hash<int> hi;
        auto h1 = hi(42);
        auto h2 = hi(42);
        expect(h1 == h2) << "hash<int> should be deterministic";

        auto h3 = hi(43);
        expect(h1 != h3) << "different ints should hash differently";

        luisa::hash<float> hf;
        auto hf1 = hf(3.14f);
        auto hf2 = hf(3.14f);
        expect(hf1 == hf2);

        luisa::hash<double> hd;
        auto hd1 = hd(2.718);
        expect(hd1 != 0u);
    };
    return 0;
}();

static inline const auto reg_hash_pointer = [] {
    "hash_pointer"_test = [] {
        int a = 1, b = 2;
        luisa::hash<int *> hp;
        auto ha = hp(&a);
        auto hb = hp(&b);
        // different addresses should produce different hashes (extremely likely)
        expect(ha != hb);

        // same address should produce same hash
        auto ha2 = hp(&a);
        expect(ha == ha2);
    };
    return 0;
}();

// ---- hash_value ----

static inline const auto reg_hash_value = [] {
    "hash_value"_test = [] {
        auto h1 = luisa::hash_value(42);
        auto h2 = luisa::hash_value(42);
        expect(h1 == h2);

        auto h3 = luisa::hash_value(42, 0u);// explicit seed
        auto h4 = luisa::hash_value(42, 1u);
        expect(h3 != h4) << "different seeds should give different hash_value";
    };
    return 0;
}();

// ---- hash_combine ----

static inline const auto reg_hash_combine_initializer_list = [] {
    "hash_combine_initializer_list"_test = [] {
        auto h1 = luisa::hash_combine({1u, 2u, 3u});
        auto h2 = luisa::hash_combine({1u, 2u, 3u});
        expect(h1 == h2) << "same inputs should produce same combined hash";

        auto h3 = luisa::hash_combine({3u, 2u, 1u});
        expect(h1 != h3) << "different order should produce different hash";

        auto h4 = luisa::hash_combine({1u, 2u, 4u});
        expect(h1 != h4) << "different values should produce different hash";
    };
    return 0;
}();

static inline const auto reg_hash_combine_span = [] {
    "hash_combine_span"_test = [] {
        uint64_t data[] = {10u, 20u, 30u};
        auto h1 = luisa::hash_combine(luisa::span<const uint64_t>{data, 3u});
        auto h2 = luisa::hash_combine(luisa::span<const uint64_t>{data, 3u});
        expect(h1 == h2);
    };
    return 0;
}();

// ---- Hash128 ----

static inline const auto reg_hash128_basic = [] {
    "hash128_basic"_test = [] {
        const char data[] = "test data for hash128";
        auto h = luisa::hash128(data, sizeof(data), luisa::hash64_default_seed);

        // Hash128 has 16 bytes of data
        expect(h.data().size() == 16u);
    };
    return 0;
}();

static inline const auto reg_hash128_deterministic = [] {
    "hash128_deterministic"_test = [] {
        const char data[] = "determinism test";
        auto h1 = luisa::hash128(data, sizeof(data), 0u);
        auto h2 = luisa::hash128(data, sizeof(data), 0u);
        expect(h1 == h2) << "hash128 must be deterministic";
    };
    return 0;
}();

static inline const auto reg_hash128_different = [] {
    "hash128_different_data"_test = [] {
        const char data1[] = "aaa";
        const char data2[] = "bbb";
        auto h1 = luisa::hash128(data1, sizeof(data1), 0u);
        auto h2 = luisa::hash128(data2, sizeof(data2), 0u);
        expect(!(h1 == h2)) << "different data should produce different Hash128";
    };
    return 0;
}();

static inline const auto reg_hash128_to_string = [] {
    "hash128_to_string"_test = [] {
        const char data[] = "to_string test";
        auto h = luisa::hash128(data, sizeof(data), 0u);
        auto s = h.to_string();
        // Should produce a non-empty hex string
        expect(!s.empty());
        // 128-bit hash = 32 hex chars
        expect(s.size() == 32u) << "Hash128::to_string should return 32 hex characters";
    };
    return 0;
}();

static inline const auto reg_hash128_equality = [] {
    "hash128_equality"_test = [] {
        const char data[] = "equality";
        auto h1 = luisa::hash128(data, sizeof(data), 42u);
        auto h2 = luisa::hash128(data, sizeof(data), 42u);
        auto h3 = luisa::hash128(data, sizeof(data), 43u);
        expect(h1 == h2);
        expect(!(h1 == h3));
    };
    return 0;
}();

// ---- string_hash ----

static inline const auto reg_string_hash = [] {
    "string_hash_basic"_test = [] {
        luisa::string_hash sh;

        auto h1 = sh("hello");
        auto h2 = sh("hello");
        expect(h1 == h2) << "same string should hash to same value";

        auto h3 = sh("world");
        expect(h1 != h3);

        // hash from luisa::string
        luisa::string s = "hello";
        auto h4 = sh(s);
        expect(h4 == h1) << "c-string and luisa::string should hash the same";

        // hash from string_view
        luisa::string_view sv = "hello";
        auto h5 = sh(sv);
        expect(h5 == h1) << "c-string and string_view should hash the same";
    };
    return 0;
}();

static inline const auto reg_default_seed = [] {
    "hash64_default_seed_is_prime"_test = [] {
        // (2^61 - 1) is a Mersenne prime
        expect(luisa::hash64_default_seed == ((1ull << 61ull) - 1ull));
    };
    return 0;
}();

int main() {}
