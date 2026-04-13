// Device-side math builtins test.
// Computes math functions on GPU, reads results back, compares against CPU reference.

#include <cmath>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include "ut/ut.hpp"
#include "test_device.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

// Result layout: each function writes to a known slot in the output buffer.
// We pack all scalar results into a single float buffer for simplicity.
enum Slot : uint {
    // unary float
    SIN,
    COS,
    TAN,
    ASIN,
    ACOS,
    ATAN,
    SINH,
    COSH,
    TANH,
    EXP,
    EXP2,
    LOG,
    LOG2,
    LOG10,
    SQRT,
    RSQRT,
    FLOOR,
    CEIL,
    ROUND,
    TRUNC,
    FRACT,
    ABS_F,
    SIGN_POS,
    SIGN_NEG,
    SIGN_ZERO,
    SATURATE_LOW,
    SATURATE_MID,
    SATURATE_HIGH,
    // binary float
    POW,
    ATAN2,
    MIN_F,
    MAX_F,
    FMA_F,
    STEP,
    COPYSIGN_POS,
    COPYSIGN_NEG,
    // clamp / lerp
    CLAMP_BELOW,
    CLAMP_IN,
    CLAMP_ABOVE,
    LERP_0,
    LERP_HALF,
    LERP_1,
    // vector ops
    DOT3,
    LENGTH3,
    LENGTH_SQ3,
    CROSS_X,
    CROSS_Y,
    CROSS_Z,
    NORMALIZE_X,
    NORMALIZE_Y,
    NORMALIZE_Z,
    DISTANCE3,
    // int ops
    ABS_I,
    MIN_I,
    MAX_I,
    CLAMP_I,
    CLZ_V,
    CTZ_V,
    POPCOUNT_V,
    // isinf / isnan  (stored as 0.0 or 1.0)
    IS_INF,
    IS_NAN,
    NOT_INF,
    NOT_NAN,
    SLOT_COUNT
};

int main(int argc, char *argv[]) {
    auto [context, device] = luisa::test::create_device(argc, argv);
    Stream stream = device.create_stream();

    Buffer<float> result_buf = device.create_buffer<float>(SLOT_COUNT);

    // ---- GPU kernel ----
    Kernel1D kernel = [&] {
        auto write = [&](uint idx, Float v) { result_buf->write(static_cast<UInt>(idx), v); };

        // --- unary float ---
        Float x = 0.5f;
        write(SIN, sin(x));
        write(COS, cos(x));
        write(TAN, tan(x));
        write(ASIN, asin(x));
        write(ACOS, acos(x));
        write(ATAN, atan(x));
        write(SINH, sinh(x));
        write(COSH, cosh(x));
        write(TANH, tanh(x));
        write(EXP, exp(x));
        write(EXP2, exp2(x));
        write(LOG, luisa::compute::log(x));
        write(LOG2, log2(x));
        write(LOG10, log10(x));
        write(SQRT, sqrt(x));
        write(RSQRT, rsqrt(x));
        write(FLOOR, floor(def(1.7f)));
        write(CEIL, ceil(def(1.3f)));
        write(ROUND, round(def(2.5f)));
        write(TRUNC, trunc(def(-1.7f)));
        write(FRACT, fract(def(3.75f)));
        write(ABS_F, abs(def(-4.5f)));
        write(SIGN_POS, sign(def(3.0f)));
        write(SIGN_NEG, sign(def(-2.0f)));
        write(SIGN_ZERO, sign(def(0.0f)));
        write(SATURATE_LOW, saturate(def(-0.5f)));
        write(SATURATE_MID, saturate(def(0.3f)));
        write(SATURATE_HIGH, saturate(def(1.5f)));

        // --- binary float ---
        write(POW, pow(def(2.0f), def(3.0f)));
        write(ATAN2, atan2(def(1.0f), def(1.0f)));
        write(MIN_F, min(def(3.0f), def(5.0f)));
        write(MAX_F, max(def(3.0f), def(5.0f)));
        write(FMA_F, fma(def(2.0f), def(3.0f), def(4.0f)));
        write(STEP, step(def(0.5f), def(0.7f)));
        write(COPYSIGN_POS, copysign(def(3.0f), def(-1.0f)));
        write(COPYSIGN_NEG, copysign(def(-3.0f), def(1.0f)));

        // --- clamp / lerp ---
        write(CLAMP_BELOW, clamp(def(-1.0f), def(0.0f), def(1.0f)));
        write(CLAMP_IN, clamp(def(0.5f), def(0.0f), def(1.0f)));
        write(CLAMP_ABOVE, clamp(def(2.0f), def(0.0f), def(1.0f)));
        write(LERP_0, lerp(def(10.0f), def(20.0f), def(0.0f)));
        write(LERP_HALF, lerp(def(10.0f), def(20.0f), def(0.5f)));
        write(LERP_1, lerp(def(10.0f), def(20.0f), def(1.0f)));

        // --- vector ops ---
        Float3 a = make_float3(1.0f, 2.0f, 3.0f);
        Float3 b = make_float3(4.0f, 5.0f, 6.0f);
        write(DOT3, dot(a, b));
        write(LENGTH3, length(a));
        write(LENGTH_SQ3, length_squared(a));
        Float3 cr = cross(a, b);
        write(CROSS_X, cr.x);
        write(CROSS_Y, cr.y);
        write(CROSS_Z, cr.z);
        Float3 n = normalize(a);
        write(NORMALIZE_X, n.x);
        write(NORMALIZE_Y, n.y);
        write(NORMALIZE_Z, n.z);
        write(DISTANCE3, distance(a, b));

        // --- int ops (cast to float for storage) ---
        write(ABS_I, cast<float>(abs(def(-7))));
        write(MIN_I, cast<float>(min(def(3), def(5))));
        write(MAX_I, cast<float>(max(def(3), def(5))));
        write(CLAMP_I, cast<float>(clamp(def(10), def(0), def(5))));
        write(CLZ_V, cast<float>(clz(def(16u))));
        write(CTZ_V, cast<float>(ctz(def(16u))));
        write(POPCOUNT_V, cast<float>(popcount(def(0b10110u))));

        // --- isinf / isnan ---
        Float inf_val = def(1.0f) / def(0.0f);
        Float nan_val = sqrt(def(-1.0f));
        Float normal_val = def(1.0f);
        write(IS_INF, ite(isinf(inf_val), 1.0f, 0.0f));
        write(IS_NAN, ite(isnan(nan_val), 1.0f, 0.0f));
        write(NOT_INF, ite(isinf(normal_val), 1.0f, 0.0f));
        write(NOT_NAN, ite(isnan(normal_val), 1.0f, 0.0f));
    };

    auto shader = device.compile(kernel);

    luisa::vector<float> results(SLOT_COUNT);
    stream << shader().dispatch(1u)
           << result_buf.copy_to(results.data())
           << synchronize();

    // ---- CPU reference + comparison ----
    auto approx = [](float a, float b, float eps = 1e-4f) {
        return std::abs(a - b) < eps;
    };

    auto check = [&](Slot s, float expected, const char *name) {
        bool ok = approx(results[s], expected);
        expect(static_cast<bool>(ok))
            << name << ": got" << results[s] << "expected" << expected;
    };

    auto check_exact = [&](Slot s, float expected, const char *name) {
        bool ok = results[s] == expected;
        expect(static_cast<bool>(ok))
            << name << ": got" << results[s] << "expected" << expected;
    };

    // unary float (x = 0.5)
    check(SIN, std::sin(0.5f), "sin");
    check(COS, std::cos(0.5f), "cos");
    check(TAN, std::tan(0.5f), "tan");
    check(ASIN, std::asin(0.5f), "asin");
    check(ACOS, std::acos(0.5f), "acos");
    check(ATAN, std::atan(0.5f), "atan");
    check(SINH, std::sinh(0.5f), "sinh");
    check(COSH, std::cosh(0.5f), "cosh");
    check(TANH, std::tanh(0.5f), "tanh");
    check(EXP, std::exp(0.5f), "exp");
    check(EXP2, std::exp2(0.5f), "exp2");
    check(LOG, std::log(0.5f), "log");
    check(LOG2, std::log2(0.5f), "log2");
    check(LOG10, std::log10(0.5f), "log10");
    check(SQRT, std::sqrt(0.5f), "sqrt");
    check(RSQRT, 1.0f / std::sqrt(0.5f), "rsqrt");
    check_exact(FLOOR, 1.0f, "floor");
    check_exact(CEIL, 2.0f, "ceil");
    check_exact(ROUND, 2.0f, "round");// round-half-to-even: round(2.5) → 2
    check_exact(TRUNC, -1.0f, "trunc");
    check(FRACT, 0.75f, "fract");
    check_exact(ABS_F, 4.5f, "abs_f");
    check_exact(SIGN_POS, 1.0f, "sign_pos");
    check_exact(SIGN_NEG, -1.0f, "sign_neg");
    check_exact(SIGN_ZERO, 0.0f, "sign_zero");
    check_exact(SATURATE_LOW, 0.0f, "saturate_low");
    check(SATURATE_MID, 0.3f, "saturate_mid");
    check_exact(SATURATE_HIGH, 1.0f, "saturate_high");

    // binary float
    check(POW, 8.0f, "pow");
    check(ATAN2, std::atan2(1.0f, 1.0f), "atan2");
    check_exact(MIN_F, 3.0f, "min_f");
    check_exact(MAX_F, 5.0f, "max_f");
    check_exact(FMA_F, 10.0f, "fma_f");
    check_exact(STEP, 1.0f, "step");
    check_exact(COPYSIGN_POS, -3.0f, "copysign_pos");
    check_exact(COPYSIGN_NEG, 3.0f, "copysign_neg");

    // clamp / lerp
    check_exact(CLAMP_BELOW, 0.0f, "clamp_below");
    check_exact(CLAMP_IN, 0.5f, "clamp_in");
    check_exact(CLAMP_ABOVE, 1.0f, "clamp_above");
    check_exact(LERP_0, 10.0f, "lerp_0");
    check_exact(LERP_HALF, 15.0f, "lerp_half");
    check_exact(LERP_1, 20.0f, "lerp_1");

    // vector ops
    // dot(1,2,3)·(4,5,6) = 4+10+18 = 32
    check_exact(DOT3, 32.0f, "dot3");
    // length(1,2,3) = sqrt(14)
    check(LENGTH3, std::sqrt(14.0f), "length3");
    check_exact(LENGTH_SQ3, 14.0f, "length_sq3");
    // cross(1,2,3)×(4,5,6) = (2*6-3*5, 3*4-1*6, 1*5-2*4) = (-3,6,-3)
    check_exact(CROSS_X, -3.0f, "cross_x");
    check_exact(CROSS_Y, 6.0f, "cross_y");
    check_exact(CROSS_Z, -3.0f, "cross_z");
    float inv_len = 1.0f / std::sqrt(14.0f);
    check(NORMALIZE_X, 1.0f * inv_len, "normalize_x");
    check(NORMALIZE_Y, 2.0f * inv_len, "normalize_y");
    check(NORMALIZE_Z, 3.0f * inv_len, "normalize_z");
    // distance = length(b-a) = length(3,3,3) = sqrt(27)
    check(DISTANCE3, std::sqrt(27.0f), "distance3");

    // int ops
    check_exact(ABS_I, 7.0f, "abs_i");
    check_exact(MIN_I, 3.0f, "min_i");
    check_exact(MAX_I, 5.0f, "max_i");
    check_exact(CLAMP_I, 5.0f, "clamp_i");
    // clz(16) = clz(0x10) = 27 on 32-bit
    check_exact(CLZ_V, 27.0f, "clz");
    // ctz(16) = 4
    check_exact(CTZ_V, 4.0f, "ctz");
    // popcount(0b10110) = 3
    check_exact(POPCOUNT_V, 3.0f, "popcount");

    // isinf / isnan
    check_exact(IS_INF, 1.0f, "is_inf");
    check_exact(IS_NAN, 1.0f, "is_nan");
    check_exact(NOT_INF, 0.0f, "not_inf");
    check_exact(NOT_NAN, 0.0f, "not_nan");

    LUISA_INFO("All device math tests passed!");
    return 0;
}
