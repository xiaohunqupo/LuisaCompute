/**
 * @file tests/for_agent/mathematics.cpp
 * @brief Test cases for luisa/core/mathematics.h
 */

#include <luisa/core/mathematics.h>
#include <luisa/core/logging.h>
#include <cmath>
#include <limits>

using namespace luisa;

// Helper for approximate equality check
template<typename T>
bool approx_eq(T a, T b, T epsilon = static_cast<T>(1e-5)) {
    return std::abs(a - b) < epsilon;
}

// Helper for vector approximate equality
template<size_t N>
bool approx_eq_vec(Vector<float, N> a, Vector<float, N> b, float epsilon = 1e-5f) {
    for (size_t i = 0; i < N; ++i) {
        if (!approx_eq(a[i], b[i], epsilon)) return false;
    }
    return true;
}

void test_next_pow2() {
    // Test uint32
    LUISA_ASSERT(next_pow2(0u) == 0u, "next_pow2(0) should be 0");
    LUISA_ASSERT(next_pow2(1u) == 1u, "next_pow2(1) should be 1");
    LUISA_ASSERT(next_pow2(2u) == 2u, "next_pow2(2) should be 2");
    LUISA_ASSERT(next_pow2(3u) == 4u, "next_pow2(3) should be 4");
    LUISA_ASSERT(next_pow2(4u) == 4u, "next_pow2(4) should be 4");
    LUISA_ASSERT(next_pow2(5u) == 8u, "next_pow2(5) should be 8");
    LUISA_ASSERT(next_pow2(1023u) == 1024u, "next_pow2(1023) should be 1024");
    LUISA_ASSERT(next_pow2(1024u) == 1024u, "next_pow2(1024) should be 1024");
    LUISA_ASSERT(next_pow2(1025u) == 2048u, "next_pow2(1025) should be 2048");
    
    // Test uint64
    LUISA_ASSERT(next_pow2(0ull) == 0ull, "next_pow2(0ull) should be 0");
    LUISA_ASSERT(next_pow2(1ull) == 1ull, "next_pow2(1ull) should be 1");
    LUISA_ASSERT(next_pow2(3ull) == 4ull, "next_pow2(3ull) should be 4");
    LUISA_ASSERT(next_pow2(1ull << 32) == (1ull << 32), "next_pow2(2^32) should be 2^32");
    LUISA_ASSERT(next_pow2((1ull << 32) + 1) == (1ull << 33), "next_pow2(2^32+1) should be 2^33");
}

void test_scalar_math() {
    // Test fract
    LUISA_ASSERT(approx_eq(fract(3.7f), 0.7f), "fract(3.7) should be 0.7");
    LUISA_ASSERT(approx_eq(fract(-3.7f), 0.3f), "fract(-3.7) should be 0.3");
    LUISA_ASSERT(approx_eq(fract(5.0f), 0.0f), "fract(5.0) should be 0.0");
    
    // Test radians/degrees conversion
    LUISA_ASSERT(approx_eq(radians(180.0f), constants::pi), "radians(180) should be pi");
    LUISA_ASSERT(approx_eq(radians(90.0f), constants::pi / 2.0f), "radians(90) should be pi/2");
    LUISA_ASSERT(approx_eq(degrees(constants::pi), 180.0f), "degrees(pi) should be 180");
    LUISA_ASSERT(approx_eq(degrees(constants::pi / 2.0f), 90.0f), "degrees(pi/2) should be 90");
    
    // Test standard math functions are available
    LUISA_ASSERT(approx_eq(sin(constants::pi / 2.0f), 1.0f), "sin(pi/2) should be 1");
    LUISA_ASSERT(approx_eq(cos(0.0f), 1.0f), "cos(0) should be 1");
    LUISA_ASSERT(approx_eq(sqrt(4.0f), 2.0f), "sqrt(4) should be 2");
    LUISA_ASSERT(approx_eq(abs(-5.0f), 5.0f), "abs(-5) should be 5");
    LUISA_ASSERT(min(3, 5) == 3, "min(3, 5) should be 3");
    LUISA_ASSERT(max(3, 5) == 5, "max(3, 5) should be 5");
}

void test_vector_unary_funcs() {
    // Test vector sin
    float2 v2 = make_float2(0.0f, constants::pi / 2.0f);
    float2 sin2 = sin(v2);
    LUISA_ASSERT(approx_eq(sin2.x, 0.0f), "sin(0) should be 0");
    LUISA_ASSERT(approx_eq(sin2.y, 1.0f), "sin(pi/2) should be 1");
    
    float3 v3 = make_float3(0.0f, constants::pi / 2.0f, constants::pi);
    float3 cos3 = cos(v3);
    LUISA_ASSERT(approx_eq(cos3.x, 1.0f), "cos(0) should be 1");
    LUISA_ASSERT(approx_eq(cos3.y, 0.0f), "cos(pi/2) should be 0");
    LUISA_ASSERT(approx_eq(cos3.z, -1.0f), "cos(pi) should be -1");
    
    float4 v4 = make_float4(1.0f, 4.0f, 9.0f, 16.0f);
    float4 sqrt4 = sqrt(v4);
    LUISA_ASSERT(approx_eq(sqrt4.x, 1.0f), "sqrt(1) should be 1");
    LUISA_ASSERT(approx_eq(sqrt4.y, 2.0f), "sqrt(4) should be 2");
    LUISA_ASSERT(approx_eq(sqrt4.z, 3.0f), "sqrt(9) should be 3");
    LUISA_ASSERT(approx_eq(sqrt4.w, 4.0f), "sqrt(16) should be 4");
    
    // Test vector abs
    float3 v3neg = make_float3(-1.0f, -2.0f, -3.0f);
    float3 abs3 = abs(v3neg);
    LUISA_ASSERT(approx_eq(abs3.x, 1.0f), "abs(-1) should be 1");
    LUISA_ASSERT(approx_eq(abs3.y, 2.0f), "abs(-2) should be 2");
    LUISA_ASSERT(approx_eq(abs3.z, 3.0f), "abs(-3) should be 3");
    
    // Test fract for vectors
    float3 v3fract = make_float3(1.5f, 2.7f, -3.3f);
    float3 fract3 = fract(v3fract);
    LUISA_ASSERT(approx_eq(fract3.x, 0.5f), "fract(1.5) should be 0.5");
    LUISA_ASSERT(approx_eq(fract3.y, 0.7f), "fract(2.7) should be 0.7");
    LUISA_ASSERT(approx_eq(fract3.z, 0.7f), "fract(-3.3) should be 0.7");
    
    // Test floor/ceil
    float2 v2floor = make_float2(1.9f, -1.1f);
    float2 floor2 = floor(v2floor);
    LUISA_ASSERT(approx_eq(floor2.x, 1.0f), "floor(1.9) should be 1");
    LUISA_ASSERT(approx_eq(floor2.y, -2.0f), "floor(-1.1) should be -2");
    
    float2 v2ceil = make_float2(1.1f, -1.9f);
    float2 ceil2 = ceil(v2ceil);
    LUISA_ASSERT(approx_eq(ceil2.x, 2.0f), "ceil(1.1) should be 2");
    LUISA_ASSERT(approx_eq(ceil2.y, -1.0f), "ceil(-1.9) should be -1");
    
    // Test radians/degrees for vectors
    float3 deg3 = make_float3(0.0f, 90.0f, 180.0f);
    float3 rad3 = radians(deg3);
    LUISA_ASSERT(approx_eq(rad3.x, 0.0f), "radians(0) should be 0");
    LUISA_ASSERT(approx_eq(rad3.y, constants::pi / 2.0f), "radians(90) should be pi/2");
    LUISA_ASSERT(approx_eq(rad3.z, constants::pi), "radians(180) should be pi");
}

void test_vector_binary_funcs() {
    // Test min/max for vectors
    float2 a2 = make_float2(1.0f, 5.0f);
    float2 b2 = make_float2(3.0f, 2.0f);
    float2 min2 = min(a2, b2);
    float2 max2 = max(a2, b2);
    LUISA_ASSERT(approx_eq(min2.x, 1.0f), "min(1, 3) should be 1");
    LUISA_ASSERT(approx_eq(min2.y, 2.0f), "min(5, 2) should be 2");
    LUISA_ASSERT(approx_eq(max2.x, 3.0f), "max(1, 3) should be 3");
    LUISA_ASSERT(approx_eq(max2.y, 5.0f), "max(5, 2) should be 5");
    
    // Test scalar-vector min/max
    float3 v3 = make_float3(1.0f, 5.0f, 3.0f);
    float3 min_s3 = min(2.0f, v3);
    float3 max_s3 = max(2.0f, v3);
    float3 min_vs3 = min(v3, 2.0f);
    float3 max_vs3 = max(v3, 2.0f);
    LUISA_ASSERT(approx_eq(min_s3.x, 1.0f), "min(2, 1) should be 1");
    LUISA_ASSERT(approx_eq(min_s3.y, 2.0f), "min(2, 5) should be 2");
    LUISA_ASSERT(approx_eq(min_s3.z, 2.0f), "min(2, 3) should be 2");
    LUISA_ASSERT(approx_eq(max_s3.x, 2.0f), "max(2, 1) should be 2");
    LUISA_ASSERT(approx_eq(max_s3.y, 5.0f), "max(2, 5) should be 5");
    LUISA_ASSERT(approx_eq(max_s3.z, 3.0f), "max(2, 3) should be 3");
    LUISA_ASSERT(approx_eq(min_vs3.x, 1.0f), "min(1, 2) should be 1");
    LUISA_ASSERT(approx_eq(max_vs3.x, 2.0f), "max(1, 2) should be 2");
    
    // Test pow for vectors
    float2 base2 = make_float2(2.0f, 3.0f);
    float2 exp2 = make_float2(3.0f, 2.0f);
    float2 pow2 = pow(base2, exp2);
    LUISA_ASSERT(approx_eq(pow2.x, 8.0f), "pow(2, 3) should be 8");
    LUISA_ASSERT(approx_eq(pow2.y, 9.0f), "pow(3, 2) should be 9");
    
    // Test atan2 for vectors
    float2 y2 = make_float2(1.0f, -1.0f);
    float2 x2 = make_float2(1.0f, 1.0f);
    float2 atan2_2 = atan2(y2, x2);
    LUISA_ASSERT(approx_eq(atan2_2.x, constants::pi / 4.0f), "atan2(1, 1) should be pi/4");
    LUISA_ASSERT(approx_eq(atan2_2.y, -constants::pi / 4.0f), "atan2(-1, 1) should be -pi/4");
    
    // Test fmod for vectors
    float2 fmod_a2 = make_float2(7.0f, 10.0f);
    float2 fmod_b2 = make_float2(3.0f, 4.0f);
    float2 fmod2 = fmod(fmod_a2, fmod_b2);
    LUISA_ASSERT(approx_eq(fmod2.x, 1.0f), "fmod(7, 3) should be 1");
    LUISA_ASSERT(approx_eq(fmod2.y, 2.0f), "fmod(10, 4) should be 2");
}

void test_isnan_isinf() {
    // Test scalar isnan/isinf
    LUISA_ASSERT(!std::isnan(1.0f), "isnan(1.0) should be false");
    LUISA_ASSERT(std::isnan(std::numeric_limits<float>::quiet_NaN()), "isnan(NaN) should be true");
    LUISA_ASSERT(!std::isinf(1.0f), "isinf(1.0) should be false");
    LUISA_ASSERT(std::isinf(std::numeric_limits<float>::infinity()), "isinf(inf) should be true");
    LUISA_ASSERT(std::isinf(-std::numeric_limits<float>::infinity()), "isinf(-inf) should be true");
    
    // Test vector isnan/isinf
    float3 nan3 = make_float3(1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f);
    bool3 isnan3 = isnan(nan3);
    LUISA_ASSERT(!isnan3.x, "isnan(1.0) should be false");
    LUISA_ASSERT(isnan3.y, "isnan(NaN) should be true");
    LUISA_ASSERT(!isnan3.z, "isnan(2.0) should be false");
    
    float4 inf4 = make_float4(1.0f, std::numeric_limits<float>::infinity(), 
                              -std::numeric_limits<float>::infinity(), 2.0f);
    bool4 isinf4 = isinf(inf4);
    LUISA_ASSERT(!isinf4.x, "isinf(1.0) should be false");
    LUISA_ASSERT(isinf4.y, "isinf(inf) should be true");
    LUISA_ASSERT(isinf4.z, "isinf(-inf) should be true");
    LUISA_ASSERT(!isinf4.w, "isinf(2.0) should be false");
}

void test_select() {
    // Test scalar select
    LUISA_ASSERT(select(0.0f, 1.0f, true) == 1.0f, "select(0, 1, true) should be 1");
    LUISA_ASSERT(select(0.0f, 1.0f, false) == 0.0f, "select(0, 1, false) should be 0");
    
    // Test vector select
    float2 f2 = make_float2(1.0f, 2.0f);
    float2 t2 = make_float2(10.0f, 20.0f);
    bool2 p2 = make_bool2(true, false);
    float2 r2 = select(f2, t2, p2);
    LUISA_ASSERT(r2.x == 10.0f, "select(1, 10, true) should be 10");
    LUISA_ASSERT(r2.y == 2.0f, "select(2, 20, false) should be 2");
    
    float3 f3 = make_float3(1.0f, 2.0f, 3.0f);
    float3 t3 = make_float3(10.0f, 20.0f, 30.0f);
    bool3 p3 = make_bool3(false, true, false);
    float3 r3 = select(f3, t3, p3);
    LUISA_ASSERT(r3.x == 1.0f, "select(1, 10, false) should be 1");
    LUISA_ASSERT(r3.y == 20.0f, "select(2, 20, true) should be 20");
    LUISA_ASSERT(r3.z == 3.0f, "select(3, 30, false) should be 3");
}

void test_lerp_clamp() {
    // Test scalar lerp
    float lerp_result = lerp(0.0f, 10.0f, 0.5f);
    LUISA_ASSERT(approx_eq(lerp_result, 5.0f), "lerp(0, 10, 0.5) should be 5");
    LUISA_ASSERT(approx_eq(lerp(0.0f, 10.0f, 0.0f), 0.0f), "lerp(0, 10, 0) should be 0");
    LUISA_ASSERT(approx_eq(lerp(0.0f, 10.0f, 1.0f), 10.0f), "lerp(0, 10, 1) should be 10");
    
    // Test vector lerp
    float2 a2 = make_float2(0.0f, 10.0f);
    float2 b2 = make_float2(10.0f, 20.0f);
    float2 lerp2 = lerp(a2, b2, 0.5f);
    LUISA_ASSERT(approx_eq(lerp2.x, 5.0f), "lerp(0, 10, 0.5) should be 5");
    LUISA_ASSERT(approx_eq(lerp2.y, 15.0f), "lerp(10, 20, 0.5) should be 15");
    
    // Test scalar clamp
    LUISA_ASSERT(clamp(5.0f, 0.0f, 10.0f) == 5.0f, "clamp(5, 0, 10) should be 5");
    LUISA_ASSERT(clamp(-5.0f, 0.0f, 10.0f) == 0.0f, "clamp(-5, 0, 10) should be 0");
    LUISA_ASSERT(clamp(15.0f, 0.0f, 10.0f) == 10.0f, "clamp(15, 0, 10) should be 10");
    
    // Test vector clamp
    float3 v3 = make_float3(-1.0f, 5.0f, 15.0f);
    float3 clamp3 = clamp(v3, 0.0f, 10.0f);
    LUISA_ASSERT(clamp3.x == 0.0f, "clamp(-1, 0, 10) should be 0");
    LUISA_ASSERT(clamp3.y == 5.0f, "clamp(5, 0, 10) should be 5");
    LUISA_ASSERT(clamp3.z == 10.0f, "clamp(15, 0, 10) should be 10");
}

void test_vector_operations() {
    // Test dot product
    float2 v2a = make_float2(1.0f, 2.0f);
    float2 v2b = make_float2(3.0f, 4.0f);
    LUISA_ASSERT(approx_eq(dot(v2a, v2b), 11.0f), "dot((1,2), (3,4)) should be 11");
    
    float3 v3a = make_float3(1.0f, 2.0f, 3.0f);
    float3 v3b = make_float3(4.0f, 5.0f, 6.0f);
    LUISA_ASSERT(approx_eq(dot(v3a, v3b), 32.0f), "dot((1,2,3), (4,5,6)) should be 32");
    
    float4 v4a = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float4 v4b = make_float4(5.0f, 6.0f, 7.0f, 8.0f);
    LUISA_ASSERT(approx_eq(dot(v4a, v4b), 70.0f), "dot((1,2,3,4), (5,6,7,8)) should be 70");
    
    // Test length
    float2 v2len = make_float2(3.0f, 4.0f);
    LUISA_ASSERT(approx_eq(length(v2len), 5.0f), "length((3,4)) should be 5");
    
    float3 v3len = make_float3(1.0f, 2.0f, 2.0f);
    LUISA_ASSERT(approx_eq(length(v3len), 3.0f), "length((1,2,2)) should be 3");
    
    // Test normalize
    float2 v2norm = make_float2(3.0f, 4.0f);
    float2 n2 = normalize(v2norm);
    LUISA_ASSERT(approx_eq(length(n2), 1.0f), "length of normalized vector should be 1");
    LUISA_ASSERT(approx_eq(n2.x, 0.6f), "normalize((3,4)).x should be 0.6");
    LUISA_ASSERT(approx_eq(n2.y, 0.8f), "normalize((3,4)).y should be 0.8");
    
    // Test distance
    float2 v2dist1 = make_float2(1.0f, 2.0f);
    float2 v2dist2 = make_float2(4.0f, 6.0f);
    LUISA_ASSERT(approx_eq(distance(v2dist1, v2dist2), 5.0f), "distance between (1,2) and (4,6) should be 5");
    
    // Test cross product
    float3 v3x = make_float3(1.0f, 0.0f, 0.0f);
    float3 v3y = make_float3(0.0f, 1.0f, 0.0f);
    float3 v3z = cross(v3x, v3y);
    LUISA_ASSERT(approx_eq(v3z.x, 0.0f), "cross(x, y).x should be 0");
    LUISA_ASSERT(approx_eq(v3z.y, 0.0f), "cross(x, y).y should be 0");
    LUISA_ASSERT(approx_eq(v3z.z, 1.0f), "cross(x, y).z should be 1");
    
    float3 v3a_cross = make_float3(1.0f, 2.0f, 3.0f);
    float3 v3b_cross = make_float3(4.0f, 5.0f, 6.0f);
    float3 cross_result = cross(v3a_cross, v3b_cross);
    // cross((1,2,3), (4,5,6)) = (2*6-3*5, 3*4-1*6, 1*5-2*4) = (-3, 6, -3)
    LUISA_ASSERT(approx_eq(cross_result.x, -3.0f), "cross((1,2,3), (4,5,6)).x should be -3");
    LUISA_ASSERT(approx_eq(cross_result.y, 6.0f), "cross((1,2,3), (4,5,6)).y should be 6");
    LUISA_ASSERT(approx_eq(cross_result.z, -3.0f), "cross((1,2,3), (4,5,6)).z should be -3");
}

void test_matrix_transpose() {
    // Test float2x2 transpose
    float2x2 m2 = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
    float2x2 t2 = transpose(m2);
    // Original: col0=(1,2), col1=(3,4)
    // Transposed: col0=(1,3), col1=(2,4)
    LUISA_ASSERT(approx_eq(t2[0][0], 1.0f), "transpose[0][0] should be 1");
    LUISA_ASSERT(approx_eq(t2[0][1], 3.0f), "transpose[0][1] should be 3");
    LUISA_ASSERT(approx_eq(t2[1][0], 2.0f), "transpose[1][0] should be 2");
    LUISA_ASSERT(approx_eq(t2[1][1], 4.0f), "transpose[1][1] should be 4");
    
    // Test float3x3 transpose
    float3x3 m3 = make_float3x3(
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f);
    float3x3 t3 = transpose(m3);
    LUISA_ASSERT(approx_eq(t3[0][0], 1.0f), "transpose[0][0] should be 1");
    LUISA_ASSERT(approx_eq(t3[0][1], 4.0f), "transpose[0][1] should be 4");
    LUISA_ASSERT(approx_eq(t3[0][2], 7.0f), "transpose[0][2] should be 7");
    LUISA_ASSERT(approx_eq(t3[1][0], 2.0f), "transpose[1][0] should be 2");
    LUISA_ASSERT(approx_eq(t3[1][1], 5.0f), "transpose[1][1] should be 5");
    LUISA_ASSERT(approx_eq(t3[1][2], 8.0f), "transpose[1][2] should be 8");
    LUISA_ASSERT(approx_eq(t3[2][0], 3.0f), "transpose[2][0] should be 3");
    LUISA_ASSERT(approx_eq(t3[2][1], 6.0f), "transpose[2][1] should be 6");
    LUISA_ASSERT(approx_eq(t3[2][2], 9.0f), "transpose[2][2] should be 9");
    
    // Test float4x4 transpose
    float4x4 m4 = make_float4x4(
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f);
    float4x4 t4 = transpose(m4);
    LUISA_ASSERT(approx_eq(t4[0][0], 1.0f), "transpose[0][0] should be 1");
    LUISA_ASSERT(approx_eq(t4[0][3], 13.0f), "transpose[0][3] should be 13");
    LUISA_ASSERT(approx_eq(t4[3][0], 4.0f), "transpose[3][0] should be 4");
    LUISA_ASSERT(approx_eq(t4[3][3], 16.0f), "transpose[3][3] should be 16");
}

void test_matrix_inverse() {
    // Test float2x2 inverse
    float2x2 m2 = make_float2x2(4.0f, 2.0f, 3.0f, 1.0f);
    float2x2 inv2 = inverse(m2);
    // det = 4*1 - 2*3 = -2
    // inv = (1/-2) * [[1, -2], [-3, 4]] = [[-0.5, 1], [1.5, -2]]
    LUISA_ASSERT(approx_eq(inv2[0][0], -0.5f), "inverse[0][0] should be -0.5");
    LUISA_ASSERT(approx_eq(inv2[0][1], 1.0f), "inverse[0][1] should be 1");
    LUISA_ASSERT(approx_eq(inv2[1][0], 1.5f), "inverse[1][0] should be 1.5");
    LUISA_ASSERT(approx_eq(inv2[1][1], -2.0f), "inverse[1][1] should be -2");
    
    // Verify: M * M^-1 = I
    float2x2 identity2 = m2 * inv2;
    LUISA_ASSERT(approx_eq(identity2[0][0], 1.0f, 1e-4f), "M * M^-1 [0][0] should be 1");
    LUISA_ASSERT(approx_eq(identity2[0][1], 0.0f, 1e-4f), "M * M^-1 [0][1] should be 0");
    LUISA_ASSERT(approx_eq(identity2[1][0], 0.0f, 1e-4f), "M * M^-1 [1][0] should be 0");
    LUISA_ASSERT(approx_eq(identity2[1][1], 1.0f, 1e-4f), "M * M^-1 [1][1] should be 1");
    
    // Test float3x3 inverse
    float3x3 m3 = make_float3x3(
        1.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 4.0f);
    float3x3 inv3 = inverse(m3);
    LUISA_ASSERT(approx_eq(inv3[0][0], 1.0f), "inverse diag[0][0] should be 1");
    LUISA_ASSERT(approx_eq(inv3[1][1], 0.5f), "inverse diag[1][1] should be 0.5");
    LUISA_ASSERT(approx_eq(inv3[2][2], 0.25f), "inverse diag[2][2] should be 0.25");
    
    // Test float4x4 inverse
    float4x4 m4 = make_float4x4(
        2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 4.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 8.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 16.0f);
    float4x4 inv4 = inverse(m4);
    LUISA_ASSERT(approx_eq(inv4[0][0], 0.5f), "inverse diag[0][0] should be 0.5");
    LUISA_ASSERT(approx_eq(inv4[1][1], 0.25f), "inverse diag[1][1] should be 0.25");
    LUISA_ASSERT(approx_eq(inv4[2][2], 0.125f), "inverse diag[2][2] should be 0.125");
    LUISA_ASSERT(approx_eq(inv4[3][3], 0.0625f), "inverse diag[3][3] should be 0.0625");
}

void test_matrix_determinant() {
    // Test float2x2 determinant
    float2x2 m2 = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
    float det2 = determinant(m2);
    // det = 1*4 - 2*3 = -2
    LUISA_ASSERT(approx_eq(det2, -2.0f), "determinant of [[1,3],[2,4]] should be -2");
    
    // Test float3x3 determinant
    float3x3 m3 = make_float3x3(
        1.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 3.0f);
    float det3 = determinant(m3);
    LUISA_ASSERT(approx_eq(det3, 6.0f), "determinant of diag(1,2,3) should be 6");
    
    // Test float4x4 determinant
    float4x4 m4 = make_float4x4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 3.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 4.0f);
    float det4 = determinant(m4);
    LUISA_ASSERT(approx_eq(det4, 24.0f), "determinant of diag(1,2,3,4) should be 24");
}

void test_transform_functions() {
    // Test translation matrix
    float4x4 trans = translation(1.0f, 2.0f, 3.0f);
    LUISA_ASSERT(approx_eq(trans[3][0], 1.0f), "translation x should be 1");
    LUISA_ASSERT(approx_eq(trans[3][1], 2.0f), "translation y should be 2");
    LUISA_ASSERT(approx_eq(trans[3][2], 3.0f), "translation z should be 3");
    LUISA_ASSERT(approx_eq(trans[3][3], 1.0f), "translation w should be 1");
    
    float4x4 trans_vec = translation(make_float3(4.0f, 5.0f, 6.0f));
    LUISA_ASSERT(approx_eq(trans_vec[3][0], 4.0f), "translation x should be 4");
    LUISA_ASSERT(approx_eq(trans_vec[3][1], 5.0f), "translation y should be 5");
    LUISA_ASSERT(approx_eq(trans_vec[3][2], 6.0f), "translation z should be 6");
    
    // Test scaling matrix
    float4x4 scale = scaling(2.0f, 3.0f, 4.0f);
    LUISA_ASSERT(approx_eq(scale[0][0], 2.0f), "scaling x should be 2");
    LUISA_ASSERT(approx_eq(scale[1][1], 3.0f), "scaling y should be 3");
    LUISA_ASSERT(approx_eq(scale[2][2], 4.0f), "scaling z should be 4");
    LUISA_ASSERT(approx_eq(scale[3][3], 1.0f), "scaling w should be 1");
    
    float4x4 scale_uni = scaling(5.0f);
    LUISA_ASSERT(approx_eq(scale_uni[0][0], 5.0f), "uniform scaling should be 5");
    LUISA_ASSERT(approx_eq(scale_uni[1][1], 5.0f), "uniform scaling should be 5");
    LUISA_ASSERT(approx_eq(scale_uni[2][2], 5.0f), "uniform scaling should be 5");
    
    // Test rotation matrix (around z-axis)
    float4x4 rot_z = rotation(make_float3(0.0f, 0.0f, 1.0f), constants::pi / 2.0f);
    float4 test_vec = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    float4 rotated = rot_z * test_vec;
    LUISA_ASSERT(approx_eq(rotated.x, 0.0f, 1e-4f), "rotation 90 deg around z should give x=0");
    LUISA_ASSERT(approx_eq(rotated.y, 1.0f, 1e-4f), "rotation 90 deg around z should give y=1");
    LUISA_ASSERT(approx_eq(rotated.z, 0.0f, 1e-4f), "rotation 90 deg around z should give z=0");
}

void test_sign() {
    // Test float sign
    LUISA_ASSERT(sign(5.0f) == 1.0f, "sign(5.0) should be 1");
    LUISA_ASSERT(sign(-3.0f) == -1.0f, "sign(-3.0) should be -1");
    LUISA_ASSERT(sign(0.0f) == 1.0f, "sign(0.0) should be 1");
    
    // Test float vector sign
    float3 v3 = make_float3(5.0f, -3.0f, 0.0f);
    float3 s3 = sign(v3);
    LUISA_ASSERT(s3.x == 1.0f, "sign(5.0) should be 1");
    LUISA_ASSERT(s3.y == -1.0f, "sign(-3.0) should be -1");
    LUISA_ASSERT(s3.z == 1.0f, "sign(0.0) should be 1");
    
    // Test double sign
    LUISA_ASSERT(sign(5.0) == 1.0, "sign(5.0) should be 1");
    LUISA_ASSERT(sign(-3.0) == -1.0, "sign(-3.0) should be -1");
    
    // Test int sign
    LUISA_ASSERT(sign(5) == 1, "sign(5) should be 1");
    LUISA_ASSERT(sign(-3) == -1, "sign(-3) should be -1");
    LUISA_ASSERT(sign(0) == 1, "sign(0) should be 1");
    
    int2 i2 = make_int2(5, -3);
    int2 si2 = sign(i2);
    LUISA_ASSERT(si2.x == 1, "sign(5) should be 1");
    LUISA_ASSERT(si2.y == -1, "sign(-3) should be -1");
}

void test_fma() {
    // Test scalar fma
    float fma_result = fma(2.0f, 3.0f, 4.0f);
    LUISA_ASSERT(approx_eq(fma_result, 10.0f), "fma(2, 3, 4) should be 2*3+4=10");
    
    // Test vector fma
    float2 a2 = make_float2(1.0f, 2.0f);
    float2 b2 = make_float2(3.0f, 4.0f);
    float2 c2 = make_float2(5.0f, 6.0f);
    float2 fma2 = fma(a2, b2, c2);
    LUISA_ASSERT(approx_eq(fma2.x, 8.0f), "fma(1, 3, 5) should be 8");
    LUISA_ASSERT(approx_eq(fma2.y, 14.0f), "fma(2, 4, 6) should be 14");
    
    float3 a3 = make_float3(1.0f, 2.0f, 3.0f);
    float3 b3 = make_float3(4.0f, 5.0f, 6.0f);
    float3 c3 = make_float3(7.0f, 8.0f, 9.0f);
    float3 fma3 = fma(a3, b3, c3);
    LUISA_ASSERT(approx_eq(fma3.x, 11.0f), "fma(1, 4, 7) should be 11");
    LUISA_ASSERT(approx_eq(fma3.y, 18.0f), "fma(2, 5, 8) should be 18");
    LUISA_ASSERT(approx_eq(fma3.z, 27.0f), "fma(3, 6, 9) should be 27");
    
    // Test mixed scalar-vector fma
    float3 fma_s3 = fma(2.0f, a3, c3);
    LUISA_ASSERT(approx_eq(fma_s3.x, 9.0f), "fma(2, 1, 7) should be 9");
    LUISA_ASSERT(approx_eq(fma_s3.y, 12.0f), "fma(2, 2, 8) should be 12");
    LUISA_ASSERT(approx_eq(fma_s3.z, 15.0f), "fma(2, 3, 9) should be 15");
}

void test_double_precision() {
    // Test double vector operations
    double2 d2a = make_double2(1.0, 2.0);
    double2 d2b = make_double2(3.0, 4.0);
    double dot2 = dot(d2a, d2b);
    LUISA_ASSERT(approx_eq(dot2, 11.0), "dot((1,2), (3,4)) should be 11");
    
    double3 d3a = make_double3(1.0, 2.0, 3.0);
    double3 d3b = make_double3(4.0, 5.0, 6.0);
    double dot3 = dot(d3a, d3b);
    LUISA_ASSERT(approx_eq(dot3, 32.0), "dot((1,2,3), (4,5,6)) should be 32");
    
    // Test double matrix transpose
    double2x2 dm2 = make_double2x2(1.0, 2.0, 3.0, 4.0);
    double2x2 dt2 = transpose(dm2);
    LUISA_ASSERT(approx_eq(dt2[0][0], 1.0), "transpose[0][0] should be 1");
    LUISA_ASSERT(approx_eq(dt2[0][1], 3.0), "transpose[0][1] should be 3");
    LUISA_ASSERT(approx_eq(dt2[1][0], 2.0), "transpose[1][0] should be 2");
    LUISA_ASSERT(approx_eq(dt2[1][1], 4.0), "transpose[1][1] should be 4");
    
    // Test double matrix inverse
    double2x2 dm2_inv = make_double2x2(4.0, 2.0, 3.0, 1.0);
    double2x2 dinv2 = inverse(dm2_inv);
    LUISA_ASSERT(approx_eq(dinv2[0][0], -0.5), "inverse[0][0] should be -0.5");
    LUISA_ASSERT(approx_eq(dinv2[0][1], 1.0), "inverse[0][1] should be 1");
    LUISA_ASSERT(approx_eq(dinv2[1][0], 1.5), "inverse[1][0] should be 1.5");
    LUISA_ASSERT(approx_eq(dinv2[1][1], -2.0), "inverse[1][1] should be -2");
    
    // Test double determinant
    double ddet2 = determinant(dm2_inv);
    LUISA_ASSERT(approx_eq(ddet2, -2.0), "determinant should be -2");
}

int main() {
    LUISA_INFO("Testing next_pow2...");
    test_next_pow2();
    
    LUISA_INFO("Testing scalar math functions...");
    test_scalar_math();
    
    LUISA_INFO("Testing vector unary functions...");
    test_vector_unary_funcs();
    
    LUISA_INFO("Testing vector binary functions...");
    test_vector_binary_funcs();
    
    LUISA_INFO("Testing isnan/isinf...");
    test_isnan_isinf();
    
    LUISA_INFO("Testing select...");
    test_select();
    
    LUISA_INFO("Testing lerp/clamp...");
    test_lerp_clamp();
    
    LUISA_INFO("Testing vector operations...");
    test_vector_operations();
    
    LUISA_INFO("Testing matrix transpose...");
    test_matrix_transpose();
    
    LUISA_INFO("Testing matrix inverse...");
    test_matrix_inverse();
    
    LUISA_INFO("Testing matrix determinant...");
    test_matrix_determinant();
    
    LUISA_INFO("Testing transform functions...");
    test_transform_functions();
    
    LUISA_INFO("Testing sign...");
    test_sign();
    
    LUISA_INFO("Testing fma...");
    test_fma();
    
    LUISA_INFO("Testing double precision...");
    test_double_precision();
    
    LUISA_INFO("All mathematics tests passed!");
    return 0;
}
