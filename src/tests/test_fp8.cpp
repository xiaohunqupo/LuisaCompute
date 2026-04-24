#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
#include <random>
#include <chrono>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>

#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

// FP8 E4M3 format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
// Layout: S EEEE MMM
struct FP8E4M3 {

    static constexpr int EXP_BITS = 4;
    static constexpr int MANT_BITS = 3;
    static constexpr int BIAS = 7;
    static constexpr int MAX_EXP = (1 << EXP_BITS) - 1;       // 15
    static constexpr int INF_NAN_EXP = MAX_EXP;               // 15
    static constexpr uint8_t MANT_MASK = (1 << MANT_BITS) - 1;// 0x07
    static constexpr uint8_t SIGN_MASK = 0x80;

    // Convert FP32 to FP8 E4M3 with round-to-nearest-even
    static uint8_t from_float(float v) {
        uint8_t bits{};
        if (std::isnan(v)) {
            bits = 0x7F;// canonical NaN
            return bits;
        }
        if (v == 0.0f) {
            bits = 0;
            return bits;
        }
        uint32_t sign = 0;
        if (v < 0.0f) {
            sign = 1;
            v = -v;
        }

        // E4M3 max finite value: E=15, M=6 -> (1 + 6/8) * 2^(15-7) = 1.75 * 256 = 448
        constexpr float max_finite = 448.0f;
        if (v > max_finite) {
            // Clamp to max finite
            bits = sign ? 0xFE : 0x7E;// E=15, M=6
            return bits;
        }

        // Decompose float
        int exp;
        float mant = std::frexp(v, &exp);// v = mant * 2^exp, mant in [0.5, 1.0)
        // Adjust to get mant in [1.0, 2.0)
        mant *= 2.0f;
        exp -= 1;

        int e = exp + BIAS;
        if (e <= 0) {
            // Denormal or underflow to zero
            if (e < -MANT_BITS) {
                bits = sign ? SIGN_MASK : 0;
                return bits;
            }
            // Denormal: shift mantissa right
            int shift = 1 - e;
            mant = mant / static_cast<float>(1 << shift);
            e = 0;
        }

        // Round mantissa to 3 bits
        float mant_q = std::floor(mant * (1 << MANT_BITS));
        float frac = mant * (1 << MANT_BITS) - mant_q;
        uint8_t m = static_cast<uint8_t>(mant_q) & MANT_MASK;

        // Round-to-nearest-even
        if (frac > 0.5f || (frac == 0.5f && (m & 1))) {
            m += 1;
            if (m > MANT_MASK) {
                m = 0;
                e += 1;
            }
        }

        // E4M3: E=15, M=7 is NaN, so max finite is E=15, M=6
        if (e >= INF_NAN_EXP) {
            e = INF_NAN_EXP;
            if (m >= MANT_MASK) {
                m = MANT_MASK - 1;// clamp to 6
            }
        }

        // Reconstruct
        if (e == 0 && m == 0) {
            bits = sign ? SIGN_MASK : 0;
            return bits;
        }

        bits = (sign ? SIGN_MASK : 0) | static_cast<uint8_t>(e << MANT_BITS) | m;
        return bits;
    }

    // Convert FP8 E4M3 to FP32
    static float to_float(uint8_t bits) {
        uint8_t sign = (bits & SIGN_MASK) >> 7;
        uint8_t e = (bits >> MANT_BITS) & ((1 << EXP_BITS) - 1);
        uint8_t m = bits & MANT_MASK;

        if (e == 0 && m == 0) {
            return sign ? -0.0f : 0.0f;
        }

        // E4M3 NaN: E=15, M=7
        if (e == INF_NAN_EXP && m == MANT_MASK) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        float value;
        if (e == 0) {
            // Denormal
            value = static_cast<float>(m) / static_cast<float>(1 << MANT_BITS) * std::pow(2.0f, 1 - BIAS);
        } else {
            // Normal
            value = (1.0f + static_cast<float>(m) / static_cast<float>(1 << MANT_BITS)) * std::pow(2.0f, e - BIAS);
        }
        return sign ? -value : value;
    }
};

// FP8 E5M2 format: 1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits
// Layout: S EEEEE MM
// E5M2 supports Inf/NaN: E=31, M=0 is Inf; E=31, M!=0 is NaN
struct FP8E5M2 {
    uint8_t bits;

    static constexpr int EXP_BITS = 5;
    static constexpr int MANT_BITS = 2;
    static constexpr int BIAS = 15;
    static constexpr int MAX_EXP = (1 << EXP_BITS) - 1;       // 31
    static constexpr int INF_NAN_EXP = MAX_EXP;               // 31
    static constexpr uint8_t MANT_MASK = (1 << MANT_BITS) - 1;// 0x03
    static constexpr uint8_t SIGN_MASK = 0x80;

    // Convert FP32 to FP8 E5M2 with round-to-nearest-even
    static uint8_t from_float(float v) {
        uint8_t bits{};
        if (std::isnan(v)) {
            bits = 0x7F;// canonical NaN
            return bits;
        }
        if (std::isinf(v)) {
            bits = (v < 0.0f) ? 0xFC : 0x7C;// E=31, M=0
            return bits;
        }
        if (v == 0.0f) {
            bits = 0;
            return bits;
        }
        uint32_t sign = 0;
        if (v < 0.0f) {
            sign = 1;
            v = -v;
        }

        // E5M2 max finite value: E=30, M=3 -> (1 + 3/4) * 2^(30-15) = 1.75 * 32768 = 57344
        constexpr float max_finite = 57344.0f;
        if (v > max_finite) {
            bits = sign ? 0xFC : 0x7C;// E=31, M=0 -> Inf
            return bits;
        }

        // Decompose float
        int exp;
        float mant = std::frexp(v, &exp);// v = mant * 2^exp, mant in [0.5, 1.0)
        // Adjust to get mant in [1.0, 2.0)
        mant *= 2.0f;
        exp -= 1;

        int e = exp + BIAS;
        if (e <= 0) {
            // Denormal or underflow to zero
            if (e < -MANT_BITS) {
                bits = sign ? SIGN_MASK : 0;
                return bits;
            }
            // Denormal: shift mantissa right
            int shift = 1 - e;
            mant = mant / static_cast<float>(1 << shift);
            e = 0;
        }

        // Round mantissa to 2 bits
        float mant_q = std::floor(mant * (1 << MANT_BITS));
        float frac = mant * (1 << MANT_BITS) - mant_q;
        uint8_t m = static_cast<uint8_t>(mant_q) & MANT_MASK;

        // Round-to-nearest-even
        if (frac > 0.5f || (frac == 0.5f && (m & 1))) {
            m += 1;
            if (m > MANT_MASK) {
                m = 0;
                e += 1;
            }
        }

        // E5M2: E=31, M=0 is Inf, any M!=0 is NaN
        if (e >= INF_NAN_EXP) {
            e = INF_NAN_EXP;
            m = 0;// Inf
        }

        // Reconstruct
        if (e == 0 && m == 0) {
            bits = sign ? SIGN_MASK : 0;
            return bits;
        }

        bits = (sign ? SIGN_MASK : 0) | static_cast<uint8_t>(e << MANT_BITS) | m;
        return bits;
    }

    // Convert FP8 E5M2 to FP32
    static float to_float(uint8_t bits) {
        uint8_t sign = (bits & SIGN_MASK) >> 7;
        uint8_t e = (bits >> MANT_BITS) & ((1 << EXP_BITS) - 1);
        uint8_t m = bits & MANT_MASK;

        if (e == 0 && m == 0) {
            return sign ? -0.0f : 0.0f;
        }

        // E5M2 Inf: E=31, M=0
        if (e == INF_NAN_EXP && m == 0) {
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        }
        // E5M2 NaN: E=31, M!=0
        if (e == INF_NAN_EXP && m != 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        float value;
        if (e == 0) {
            // Denormal
            value = static_cast<float>(m) / static_cast<float>(1 << MANT_BITS) * std::pow(2.0f, 1 - BIAS);
        } else {
            // Normal
            value = (1.0f + static_cast<float>(m) / static_cast<float>(1 << MANT_BITS)) * std::pow(2.0f, e - BIAS);
        }
        return sign ? -value : value;
    }
};

// ==================== DSL Callables ====================

Callable<uint(half)> fp8e4m3_from_float() {
    static Callable _c{[](Half v) noexcept {
        $if (luisa::compute::dsl::isnan(v)) {
            $return(0x7Fu);
        };
        $if (v == half(0.0f)) {
            $return(0u);
        };

        UInt sign = 0u;
        $if (v < half(0.0f)) {
            sign = 1u;
            v = -v;
        };

        $if (v > half(448.0f)) {
            $return(ite(sign == 1u, 0xFEu, 0x7Eu));
        };

        // Decompose float: v = mant * 2^exp, mant in [1.0, 2.0)
        Int exp = floor(log2(v)).cast<int>();
        Half mant = v / pow(half(2.0f), exp.cast<half>());
        $if (mant >= half(2.0f)) {
            mant = mant * half(0.5f);
            exp = exp + 1;
        };
        $if (mant < half(1.0f)) {
            mant = mant * half(2.0f);
            exp = exp - 1;
        };

        Int e = exp + 7;// BIAS
        $if (e <= 0) {
            $if (e < -3) {// -MANT_BITS
                $return(ite(sign == 1u, 0x80u, 0u));
            };
            Int shift = 1 - e;
            mant = mant / pow(half(2.0f), shift.cast<half>());
            e = 0;
        };

        // Round mantissa to 3 bits
        Half mant_scaled = mant * half(8.0f);
        Half mant_q = floor(mant_scaled);
        Half frac = mant_scaled - mant_q;
        UInt m = mant_q.cast<uint>() & 0x07u;

        // Round-to-nearest-even
        $if (frac > half(0.5f) | (frac == half(0.5f) & (m & 1u) != 0u)) {
            m = m + 1u;
            $if (m > 0x07u) {
                m = 0u;
                e = e + 1;
            };
        };

        // E4M3: E=15, M=7 is NaN, so max finite is E=15, M=6
        $if (e >= 15) {
            e = 15;
            $if (m >= 0x07u) {
                m = 0x06u;
            };
        };

        $if (e == 0 & m == 0u) {
            $return(ite(sign == 1u, 0x80u, 0u));
        };

        UInt bits = (sign << 7u) | (e.cast<uint>() << 3u) | m;
        return bits;
    }};
    return _c;
}

Callable<half(uint)> fp8e4m3_to_float() {
    static Callable _c{[](UInt bits) noexcept {
        UInt sign = (bits >> 7u) & 1u;
        UInt e = (bits >> 3u) & 0x0Fu;
        UInt m = bits & 0x07u;

        $if (e == 0u & m == 0u) {
            $return(ite(sign != 0u, -half(0.0f), half(0.0f)));
        };

        // E4M3 NaN: E=15, M=7
        $if (e == 15u & m == 0x07u) {
            UInt nan_bits = 0x7FC00000u;
            $return(cast<half>(as<float>(nan_bits)));
        };

        Half value;
        $if (e == 0u) {
            // Denormal
            value = (m.cast<half>() / half(8.0f)) * pow(half(2.0f), half(1.0f) - half(7.0f));
        }
        $else {
            // Normal
            value = (half(1.0f) + m.cast<half>() / half(8.0f)) * pow(half(2.0f), e.cast<half>() - half(7.0f));
        };
        return ite(sign != 0u, -value, value);
    }};
    return _c;
}

Callable<uint(half)> fp8e5m2_from_float() {
    static Callable _c{[](Half v) noexcept {
        $if (luisa::compute::dsl::isnan(v)) {
            $return(0x7Fu);
        };
        $if (luisa::compute::dsl::isinf(v)) {
            $return(ite(v < half(0.0f), 0xFCu, 0x7Cu));
        };
        $if (v == half(0.0f)) {
            $return(0u);
        };

        UInt sign = 0u;
        $if (v < half(0.0f)) {
            sign = 1u;
            v = -v;
        };

        $if (v > half(57344.0f)) {
            $return(ite(sign == 1u, 0xFCu, 0x7Cu));
        };

        // Decompose float: v = mant * 2^exp, mant in [1.0, 2.0)
        Int exp = floor(log2(v)).cast<int>();
        Half mant = v / pow(half(2.0f), exp.cast<half>());
        $if (mant >= half(2.0f)) {
            mant = mant * half(0.5f);
            exp = exp + 1;
        };
        $if (mant < half(1.0f)) {
            mant = mant * half(2.0f);
            exp = exp - 1;
        };

        Int e = exp + 15;// BIAS
        $if (e <= 0) {
            $if (e < -2) {// -MANT_BITS
                $return(ite(sign == 1u, 0x80u, 0u));
            };
            Int shift = 1 - e;
            mant = mant / pow(half(2.0f), shift.cast<half>());
            e = 0;
        };

        // Round mantissa to 2 bits
        Half mant_scaled = mant * half(4.0f);
        Half mant_q = floor(mant_scaled);
        Half frac = mant_scaled - mant_q;
        UInt m = mant_q.cast<uint>() & 0x03u;

        // Round-to-nearest-even
        $if (frac > half(0.5f) | (frac == half(0.5f) & (m & 1u) != 0u)) {
            m = m + 1u;
            $if (m > 0x03u) {
                m = 0u;
                e = e + 1;
            };
        };

        // E5M2: E=31, M=0 is Inf, any M!=0 is NaN
        $if (e >= 31) {
            e = 31;
            m = 0u;
        };

        $if (e == 0 & m == 0u) {
            $return(ite(sign == 1u, 0x80u, 0u));
        };

        UInt bits = (sign << 7u) | (e.cast<uint>() << 2u) | m;
        return bits;
    }};
    return _c;
}

Callable<half(uint)> fp8e5m2_to_float() {
    static Callable _c{[](UInt bits) noexcept {
        UInt sign = (bits >> 7u) & 1u;
        UInt e = (bits >> 2u) & 0x1Fu;
        UInt m = bits & 0x03u;

        $if (e == 0u & m == 0u) {
            $return(ite(sign != 0u, -half(0.0f), half(0.0f)));
        };

        // E5M2 Inf: E=31, M=0
        $if (e == 31u & m == 0u) {
            UInt inf_bits = ite(sign != 0u, 0xFF800000u, 0x7F800000u);
            $return(cast<half>(as<float>(inf_bits)));
        };

        // E5M2 NaN: E=31, M!=0
        $if (e == 31u & m != 0u) {
            UInt nan_bits = 0x7FC00000u;
            $return(cast<half>(as<float>(nan_bits)));
        };

        Half value;
        $if (e == 0u) {
            // Denormal
            value = (m.cast<half>() / half(4.0f)) * pow(half(2.0f), half(1.0f) - half(15.0f));
        }
        $else {
            // Normal
            value = (half(1.0f) + m.cast<half>() / half(4.0f)) * pow(half(2.0f), e.cast<half>() - half(15.0f));
        };
        return ite(sign != 0u, -value, value);
    }};
    return _c;
}

// ==================== Test Kernel ====================
// Regular FP32 compute kernel for comparison
Kernel2D fp32_compute_kernel = [](BufferVar<uint> input_buf,
                                  BufferVar<float> out_e4m3_fp32,
                                  BufferVar<float> out_e5m2_fp32) noexcept {
    set_block_size(16, 16, 1);
    auto idx = dispatch_id().x + dispatch_id().y * dispatch_size().x;
    auto v = input_buf.read(idx).cast<float>();
    out_e4m3_fp32.write(idx, v);
    out_e5m2_fp32.write(idx, v * v);
};

Kernel2D fp8_test_kernel = [](BufferVar<uint> input_buf,
                              BufferVar<uint> out_to) noexcept {
    set_block_size(16, 16, 1);
    auto input_idx = dispatch_id().x + dispatch_id().y * dispatch_size().x;
    auto idx = input_idx * 4u;
    auto N = dispatch_size().x * dispatch_size().y * 4;

    UInt int_val = input_buf.read(input_idx);
    UInt result = 0;
    for (auto i : dynamic_range(4)) {
        auto bits = (int_val >> (i * 8)) & 255u;
        auto v = fp8e4m3_to_float()(bits);
        v *= v;
        result <<= 8u;
        result |= fp8e4m3_from_float()(v);
    }
    out_to.write(idx, result);
};

// ==================== Main ====================

int main(int argc, char **argv) {
    constexpr size_t N_SIZE = 4096;
    constexpr size_t N = N_SIZE * N_SIZE;
    constexpr int WARMUP_ITERS = 10;
    constexpr int PROFILE_ITERS = 128;

    // Generate random test values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> test_values(N);
    for (size_t i = 0; i < N; ++i) {
        test_values[i] = dist(rng);
    }

    // CPU reference results
    std::vector<uint8_t> cpu_e4m3_from(N);
    std::vector<uint8_t> cpu_e5m2_from(N);
    std::vector<float> cpu_e4m3_to(N);
    std::vector<float> cpu_e5m2_to(N);
    for (size_t i = 0; i < N; ++i) {
        cpu_e4m3_from[i] = FP8E4M3::from_float(test_values[i]);
        cpu_e5m2_from[i] = FP8E5M2::from_float(test_values[i]);
        cpu_e4m3_to[i] = FP8E4M3::to_float(cpu_e4m3_from[i]);
        cpu_e5m2_to[i] = FP8E5M2::to_float(cpu_e5m2_from[i]);
    }

    // GPU setup
    Context context{argv[0]};
    luisa::string backend_name = "dx";
    Device device = context.create_device(backend_name);
    Stream stream = device.create_stream(StreamTag::COMPUTE);

    auto input_buf = device.create_buffer<uint>(N);
    auto out_to = device.create_buffer<uint>(2 * N);
    auto out_fp32_e4m3 = device.create_buffer<float>(N);
    auto out_fp32_e5m2 = device.create_buffer<float>(N);

    // Upload test data
    stream << input_buf.copy_from(eastl::span{test_values.data(), test_values.size()})
           << synchronize();

    auto fp8_shader = device.compile(fp8_test_kernel);
    auto fp32_shader = device.compile(fp32_compute_kernel);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        stream << fp8_shader(input_buf, out_to).dispatch(N_SIZE, N_SIZE / 4u)
               << synchronize();
        stream << fp32_shader(input_buf, out_fp32_e4m3, out_fp32_e5m2).dispatch(N_SIZE, N_SIZE)
               << synchronize();
    }

    // Profile FP8 kernel
    auto t0 = std::chrono::high_resolution_clock::now();
    CommandList cmdlist;
    for (int i = 0; i < PROFILE_ITERS; ++i) {
        cmdlist << fp8_shader(input_buf, out_to).dispatch(N_SIZE, N_SIZE / 4u);
    }
    stream << cmdlist.commit() << synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double fp8_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / PROFILE_ITERS;

    // Profile FP32 kernel
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < PROFILE_ITERS; ++i) {
        cmdlist << fp32_shader(input_buf, out_fp32_e4m3, out_fp32_e5m2).dispatch(N_SIZE, N_SIZE);
    }
    stream << cmdlist.commit() << synchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / PROFILE_ITERS;

    LUISA_INFO("========================================");
    LUISA_INFO("Profile Results (N={}):", N);
    LUISA_INFO("  FP8  kernel avg: {} ms", fp8_ms);
    LUISA_INFO("  FP32 kernel avg: {} ms", fp32_ms);
    if (fp8_ms > 0.0) {
        LUISA_INFO("  FP32 / FP8 ratio: {}", fp32_ms / fp8_ms);
    }
    LUISA_INFO("========================================");
}
