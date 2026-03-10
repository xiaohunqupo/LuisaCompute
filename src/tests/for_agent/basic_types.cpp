// Test for basic_types.h
//
// This test verifies vector and matrix types, their constructors,
// operators, and utility functions defined in luisa/core/basic_types.h
//

#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include <limits>

using namespace luisa;

// Helper to check float equality with epsilon
bool float_eq(float a, float b, float eps = std::numeric_limits<float>::epsilon() * 100) {
    return abs(a - b) <= eps;
}

// Helper to check double equality with epsilon
bool double_eq(double a, double b, double eps = std::numeric_limits<double>::epsilon() * 100) {
    return abs(a - b) <= eps;
}

void test_vector_construction() {
    // Test single value construction
    float2 f2(1.0f);
    LUISA_ASSERT(f2.x == 1.0f && f2.y == 1.0f, "float2 scalar construction failed");

    float3 f3(2.0f);
    LUISA_ASSERT(f3.x == 2.0f && f3.y == 2.0f && f3.z == 2.0f, "float3 scalar construction failed");

    float4 f4(3.0f);
    LUISA_ASSERT(f4.x == 3.0f && f4.y == 3.0f && f4.z == 3.0f && f4.w == 3.0f, "float4 scalar construction failed");

    // Test component-wise construction
    int2 i2(1, 2);
    LUISA_ASSERT(i2.x == 1 && i2.y == 2, "int2 component construction failed");

    int3 i3(1, 2, 3);
    LUISA_ASSERT(i3.x == 1 && i3.y == 2 && i3.z == 3, "int3 component construction failed");

    int4 i4(1, 2, 3, 4);
    LUISA_ASSERT(i4.x == 1 && i4.y == 2 && i4.z == 3 && i4.w == 4, "int4 component construction failed");

    // Test zero() and one()
    auto z2 = float2::zero();
    LUISA_ASSERT(z2.x == 0.0f && z2.y == 0.0f, "float2::zero() failed");

    auto o2 = float2::one();
    LUISA_ASSERT(o2.x == 1.0f && o2.y == 1.0f, "float2::one() failed");

    auto z4 = int4::zero();
    LUISA_ASSERT(z4.x == 0 && z4.y == 0 && z4.z == 0 && z4.w == 0, "int4::zero() failed");

    auto o4 = uint4::one();
    LUISA_ASSERT(o4.x == 1 && o4.y == 1 && o4.z == 1 && o4.w == 1, "uint4::one() failed");

    LUISA_INFO("test_vector_construction passed");
}

void test_vector_element_access() {
    float3 v(1.0f, 2.0f, 3.0f);
    
    // Test operator[]
    LUISA_ASSERT(v[0] == 1.0f && v[1] == 2.0f && v[2] == 3.0f, "float3 operator[] failed");
    
    // Test modification through operator[]
    v[1] = 5.0f;
    LUISA_ASSERT(v.y == 5.0f, "float3 operator[] modification failed");

    // Test const access
    const float4 cv(1.0f, 2.0f, 3.0f, 4.0f);
    LUISA_ASSERT(cv[0] == 1.0f && cv[3] == 4.0f, "float4 const operator[] failed");

    LUISA_INFO("test_vector_element_access passed");
}

void test_vector_unary_operators() {
    // Test unary minus
    float2 f2(1.0f, -2.0f);
    float2 nf2 = -f2;
    LUISA_ASSERT(nf2.x == -1.0f && nf2.y == 2.0f, "float2 unary minus failed");

    float3 f3(1.0f, 2.0f, -3.0f);
    float3 nf3 = -f3;
    LUISA_ASSERT(nf3.x == -1.0f && nf3.y == -2.0f && nf3.z == 3.0f, "float3 unary minus failed");

    // Test unary plus
    float2 pf2 = +f2;
    LUISA_ASSERT(pf2.x == 1.0f && pf2.y == -2.0f, "float2 unary plus failed");

    // Test unary not (!)
    bool2 b2(true, false);
    bool2 nb2 = !b2;
    LUISA_ASSERT(nb2.x == false && nb2.y == true, "bool2 unary ! failed");

    bool3 b3(true, false, true);
    bool3 nb3 = !b3;
    LUISA_ASSERT(nb3.x == false && nb3.y == true && nb3.z == false, "bool3 unary ! failed");

    // Test unary xor (~) for integral types
    int2 i2(0x0F, 0xF0);
    int2 xi2 = ~i2;
    LUISA_ASSERT(xi2.x == ~0x0F && xi2.y == ~0xF0, "int2 unary ~ failed");

    LUISA_INFO("test_vector_unary_operators passed");
}

void test_vector_binary_operators() {
    // Test arithmetic operators
    float2 a(1.0f, 2.0f);
    float2 b(3.0f, 4.0f);

    float2 sum = a + b;
    LUISA_ASSERT(float_eq(sum.x, 4.0f) && float_eq(sum.y, 6.0f), "float2 addition failed");

    float2 diff = b - a;
    LUISA_ASSERT(float_eq(diff.x, 2.0f) && float_eq(diff.y, 2.0f), "float2 subtraction failed");

    float2 prod = a * b;
    LUISA_ASSERT(float_eq(prod.x, 3.0f) && float_eq(prod.y, 8.0f), "float2 multiplication failed");

    float2 quot = b / a;
    LUISA_ASSERT(float_eq(quot.x, 3.0f) && float_eq(quot.y, 2.0f), "float2 division failed");

    // Test scalar operators
    float2 scaled = a * 2.0f;
    LUISA_ASSERT(float_eq(scaled.x, 2.0f) && float_eq(scaled.y, 4.0f), "float2 scalar multiplication failed");

    float2 scaled2 = 3.0f * a;
    LUISA_ASSERT(float_eq(scaled2.x, 3.0f) && float_eq(scaled2.y, 6.0f), "scalar float2 multiplication failed");

    // Test integral operators
    int3 i1(10, 20, 30);
    int3 i2(3, 5, 7);

    int3 mod = i1 % i2;
    LUISA_ASSERT(mod.x == 1 && mod.y == 0 && mod.z == 2, "int3 modulo failed");

    int3 shifted = i1 << 1;
    LUISA_ASSERT(shifted.x == 20 && shifted.y == 40 && shifted.z == 60, "int3 left shift failed");

    int3 shifted_r = i1 >> 1;
    LUISA_ASSERT(shifted_r.x == 5 && shifted_r.y == 10 && shifted_r.z == 15, "int3 right shift failed");

    // Test bitwise operators
    uint2 u1(0x0F, 0xF0);
    uint2 u2(0xFF, 0x00);

    uint2 uor = u1 | u2;
    LUISA_ASSERT(uor.x == 0xFF && uor.y == 0xF0, "uint2 bitwise OR failed");

    uint2 uand = u1 & u2;
    LUISA_ASSERT(uand.x == 0x0F && uand.y == 0x00, "uint2 bitwise AND failed");

    uint2 uxor = u1 ^ u2;
    LUISA_ASSERT(uxor.x == 0xF0 && uxor.y == 0xF0, "uint2 bitwise XOR failed");

    LUISA_INFO("test_vector_binary_operators passed");
}

void test_vector_logic_operators() {
    // Test comparison operators
    float2 a(1.0f, 2.0f);
    float2 b(1.0f, 3.0f);

    bool2 eq = (a == b);
    LUISA_ASSERT(eq.x == true && eq.y == false, "float2 == failed");

    bool2 neq = (a != b);
    LUISA_ASSERT(neq.x == false && neq.y == true, "float2 != failed");

    bool2 lt = (a < b);
    LUISA_ASSERT(lt.x == false && lt.y == true, "float2 < failed");

    bool2 gt = (b > a);
    LUISA_ASSERT(gt.x == false && gt.y == true, "float2 > failed");

    bool2 le = (a <= b);
    LUISA_ASSERT(le.x == true && le.y == true, "float2 <= failed");

    bool2 ge = (b >= a);
    LUISA_ASSERT(ge.x == true && ge.y == true, "float2 >= failed");

    // Test boolean logic operators
    bool2 b1(true, false);
    bool2 b2(true, true);

    bool2 bor = b1 || b2;
    LUISA_ASSERT(bor.x == true && bor.y == true, "bool2 || failed");

    bool2 band = b1 && b2;
    LUISA_ASSERT(band.x == true && band.y == false, "bool2 && failed");

    LUISA_INFO("test_vector_logic_operators passed");
}

void test_vector_assignment_operators() {
    float2 a(1.0f, 2.0f);
    float2 b(3.0f, 4.0f);

    a += b;
    LUISA_ASSERT(float_eq(a.x, 4.0f) && float_eq(a.y, 6.0f), "float2 += failed");

    a -= b;
    LUISA_ASSERT(float_eq(a.x, 1.0f) && float_eq(a.y, 2.0f), "float2 -= failed");

    a *= b;
    LUISA_ASSERT(float_eq(a.x, 3.0f) && float_eq(a.y, 8.0f), "float2 *= failed");

    a /= b;
    LUISA_ASSERT(float_eq(a.x, 1.0f) && float_eq(a.y, 2.0f), "float2 /= failed");

    // Test integral assignment operators
    int2 i(10, 20);
    int2 j(3, 5);

    i %= j;
    LUISA_ASSERT(i.x == 1 && i.y == 0, "int2 %= failed");

    i = int2(4, 8);
    i <<= 1;
    LUISA_ASSERT(i.x == 8 && i.y == 16, "int2 <<= failed");

    i >>= 2;
    LUISA_ASSERT(i.x == 2 && i.y == 4, "int2 >>= failed");

    uint2 u(0x0F, 0xF0);
    u |= uint2(0xF0, 0x0F);
    LUISA_ASSERT(u.x == 0xFF && u.y == 0xFF, "uint2 |= failed");

    u = uint2(0xFF, 0xFF);
    u &= uint2(0x0F, 0xF0);
    LUISA_ASSERT(u.x == 0x0F && u.y == 0xF0, "uint2 &= failed");

    u = uint2(0xFF, 0x00);
    u ^= uint2(0x0F, 0xF0);
    LUISA_ASSERT(u.x == 0xF0 && u.y == 0xF0, "uint2 ^= failed");

    LUISA_INFO("test_vector_assignment_operators passed");
}

void test_bool_vector_functions() {
    bool2 b2_true(true, true);
    bool2 b2_mixed(true, false);
    bool2 b2_false(false, false);

    LUISA_ASSERT(any(b2_true) == true, "any(bool2 all true) failed");
    LUISA_ASSERT(any(b2_mixed) == true, "any(bool2 mixed) failed");
    LUISA_ASSERT(any(b2_false) == false, "any(bool2 all false) failed");

    LUISA_ASSERT(all(b2_true) == true, "all(bool2 all true) failed");
    LUISA_ASSERT(all(b2_mixed) == false, "all(bool2 mixed) failed");
    LUISA_ASSERT(all(b2_false) == false, "all(bool2 all false) failed");

    LUISA_ASSERT(none(b2_true) == false, "none(bool2 all true) failed");
    LUISA_ASSERT(none(b2_mixed) == false, "none(bool2 mixed) failed");
    LUISA_ASSERT(none(b2_false) == true, "none(bool2 all false) failed");

    bool3 b3_true(true, true, true);
    bool3 b3_mixed(true, false, true);
    bool3 b3_false(false, false, false);

    LUISA_ASSERT(any(b3_true) == true, "any(bool3 all true) failed");
    LUISA_ASSERT(any(b3_mixed) == true, "any(bool3 mixed) failed");
    LUISA_ASSERT(any(b3_false) == false, "any(bool3 all false) failed");

    LUISA_ASSERT(all(b3_true) == true, "all(bool3 all true) failed");
    LUISA_ASSERT(all(b3_mixed) == false, "all(bool3 mixed) failed");
    LUISA_ASSERT(all(b3_false) == false, "all(bool3 all false) failed");

    bool4 b4_true(true, true, true, true);
    bool4 b4_mixed(true, false, true, false);
    bool4 b4_false(false, false, false, false);

    LUISA_ASSERT(any(b4_true) == true, "any(bool4 all true) failed");
    LUISA_ASSERT(any(b4_mixed) == true, "any(bool4 mixed) failed");
    LUISA_ASSERT(any(b4_false) == false, "any(bool4 all false) failed");

    LUISA_ASSERT(all(b4_true) == true, "all(bool4 all true) failed");
    LUISA_ASSERT(all(b4_mixed) == false, "all(bool4 mixed) failed");
    LUISA_ASSERT(all(b4_false) == false, "all(bool4 all false) failed");

    LUISA_INFO("test_bool_vector_functions passed");
}

void test_matrix_construction() {
    // Test default construction (identity matrix)
    float2x2 m2;
    LUISA_ASSERT(m2[0][0] == 1.0f && m2[0][1] == 0.0f, "float2x2 default col 0 failed");
    LUISA_ASSERT(m2[1][0] == 0.0f && m2[1][1] == 1.0f, "float2x2 default col 1 failed");

    float3x3 m3;
    LUISA_ASSERT(m3[0][0] == 1.0f && m3[1][1] == 1.0f && m3[2][2] == 1.0f, "float3x3 default diagonal failed");
    LUISA_ASSERT(m3[0][1] == 0.0f && m3[1][0] == 0.0f, "float3x3 default off-diagonal failed");

    float4x4 m4;
    LUISA_ASSERT(m4[0][0] == 1.0f && m4[1][1] == 1.0f && m4[2][2] == 1.0f && m4[3][3] == 1.0f, "float4x4 default diagonal failed");

    // Test column vector construction
    float2x2 m2c(float2(1.0f, 2.0f), float2(3.0f, 4.0f));
    LUISA_ASSERT(m2c[0][0] == 1.0f && m2c[0][1] == 2.0f, "float2x2 col 0 failed");
    LUISA_ASSERT(m2c[1][0] == 3.0f && m2c[1][1] == 4.0f, "float2x2 col 1 failed");

    // Test eye() and fill()
    float2x2 m2eye = float2x2::eye(2.0f);
    LUISA_ASSERT(m2eye[0][0] == 2.0f && m2eye[1][1] == 2.0f, "float2x2::eye failed");
    LUISA_ASSERT(m2eye[0][1] == 0.0f && m2eye[1][0] == 0.0f, "float2x2::eye off-diagonal failed");

    float2x2 m2fill = float2x2::fill(3.0f);
    LUISA_ASSERT(m2fill[0][0] == 3.0f && m2fill[0][1] == 3.0f, "float2x2::fill col 0 failed");
    LUISA_ASSERT(m2fill[1][0] == 3.0f && m2fill[1][1] == 3.0f, "float2x2::fill col 1 failed");

    LUISA_INFO("test_matrix_construction passed");
}

void test_matrix_element_access() {
    float3x3 m(float3(1.0f, 2.0f, 3.0f), float3(4.0f, 5.0f, 6.0f), float3(7.0f, 8.0f, 9.0f));

    // Test column access
    LUISA_ASSERT(m[0][0] == 1.0f && m[0][1] == 2.0f && m[0][2] == 3.0f, "float3x3 col 0 access failed");
    LUISA_ASSERT(m[1][0] == 4.0f && m[1][1] == 5.0f && m[1][2] == 6.0f, "float3x3 col 1 access failed");
    LUISA_ASSERT(m[2][0] == 7.0f && m[2][1] == 8.0f && m[2][2] == 9.0f, "float3x3 col 2 access failed");

    // Test column modification
    m[1] = float3(10.0f, 11.0f, 12.0f);
    LUISA_ASSERT(m[1][0] == 10.0f && m[1][1] == 11.0f && m[1][2] == 12.0f, "float3x3 col modification failed");

    LUISA_INFO("test_matrix_element_access passed");
}

void test_matrix_scalar_operators() {
    float2x2 m(float2(1.0f, 2.0f), float2(3.0f, 4.0f));

    // Test scalar multiplication
    float2x2 scaled = m * 2.0f;
    LUISA_ASSERT(float_eq(scaled[0][0], 2.0f) && float_eq(scaled[0][1], 4.0f), "float2x2 * scalar col 0 failed");
    LUISA_ASSERT(float_eq(scaled[1][0], 6.0f) && float_eq(scaled[1][1], 8.0f), "float2x2 * scalar col 1 failed");

    float2x2 scaled2 = 3.0f * m;
    LUISA_ASSERT(float_eq(scaled2[0][0], 3.0f) && float_eq(scaled2[0][1], 6.0f), "scalar * float2x2 col 0 failed");
    LUISA_ASSERT(float_eq(scaled2[1][0], 9.0f) && float_eq(scaled2[1][1], 12.0f), "scalar * float2x2 col 1 failed");

    // Test scalar division
    float2x2 divided = m / 2.0f;
    LUISA_ASSERT(float_eq(divided[0][0], 0.5f) && float_eq(divided[0][1], 1.0f), "float2x2 / scalar col 0 failed");
    LUISA_ASSERT(float_eq(divided[1][0], 1.5f) && float_eq(divided[1][1], 2.0f), "float2x2 / scalar col 1 failed");

    LUISA_INFO("test_matrix_scalar_operators passed");
}

void test_matrix_vector_multiplication() {
    // Test float2x2 * float2
    float2x2 m2(float2(1.0f, 2.0f), float2(3.0f, 4.0f));
    float2 v2(1.0f, 1.0f);
    float2 r2 = m2 * v2;
    // m2 * v2 = v2.x * m2[0] + v2.y * m2[1] = 1 * (1,2) + 1 * (3,4) = (4,6)
    LUISA_ASSERT(float_eq(r2.x, 4.0f) && float_eq(r2.y, 6.0f), "float2x2 * float2 failed");

    // Test float3x3 * float3
    float3x3 m3(float3(1.0f, 0.0f, 0.0f), float3(0.0f, 2.0f, 0.0f), float3(0.0f, 0.0f, 3.0f));
    float3 v3(1.0f, 1.0f, 1.0f);
    float3 r3 = m3 * v3;
    LUISA_ASSERT(float_eq(r3.x, 1.0f) && float_eq(r3.y, 2.0f) && float_eq(r3.z, 3.0f), "float3x3 * float3 failed");

    // Test float4x4 * float4
    float4x4 m4(float4(1.0f, 0.0f, 0.0f, 0.0f), float4(0.0f, 2.0f, 0.0f, 0.0f), 
                float4(0.0f, 0.0f, 3.0f, 0.0f), float4(0.0f, 0.0f, 0.0f, 4.0f));
    float4 v4(1.0f, 1.0f, 1.0f, 1.0f);
    float4 r4 = m4 * v4;
    LUISA_ASSERT(float_eq(r4.x, 1.0f) && float_eq(r4.y, 2.0f) && float_eq(r4.z, 3.0f) && float_eq(r4.w, 4.0f), "float4x4 * float4 failed");

    LUISA_INFO("test_matrix_vector_multiplication passed");
}

void test_matrix_matrix_multiplication() {
    // Test float2x2 * float2x2
    float2x2 a(float2(1.0f, 2.0f), float2(3.0f, 4.0f));
    float2x2 b(float2(5.0f, 6.0f), float2(7.0f, 8.0f));
    float2x2 c = a * b;
    // a * b = [a * b[0], a * b[1]]
    // a * b[0] = 5*(1,2) + 6*(3,4) = (5+18, 10+24) = (23, 34)
    // a * b[1] = 7*(1,2) + 8*(3,4) = (7+24, 14+32) = (31, 46)
    LUISA_ASSERT(float_eq(c[0][0], 23.0f) && float_eq(c[0][1], 34.0f), "float2x2 * float2x2 col 0 failed");
    LUISA_ASSERT(float_eq(c[1][0], 31.0f) && float_eq(c[1][1], 46.0f), "float2x2 * float2x2 col 1 failed");

    // Test identity multiplication
    float2x2 i;
    float2x2 r = a * i;
    LUISA_ASSERT(float_eq(r[0][0], a[0][0]) && float_eq(r[0][1], a[0][1]), "float2x2 * identity col 0 failed");
    LUISA_ASSERT(float_eq(r[1][0], a[1][0]) && float_eq(r[1][1], a[1][1]), "float2x2 * identity col 1 failed");

    LUISA_INFO("test_matrix_matrix_multiplication passed");
}

void test_matrix_addition_subtraction() {
    float2x2 a(float2(1.0f, 2.0f), float2(3.0f, 4.0f));
    float2x2 b(float2(5.0f, 6.0f), float2(7.0f, 8.0f));

    float2x2 sum = a + b;
    LUISA_ASSERT(float_eq(sum[0][0], 6.0f) && float_eq(sum[0][1], 8.0f), "float2x2 + col 0 failed");
    LUISA_ASSERT(float_eq(sum[1][0], 10.0f) && float_eq(sum[1][1], 12.0f), "float2x2 + col 1 failed");

    float2x2 diff = b - a;
    LUISA_ASSERT(float_eq(diff[0][0], 4.0f) && float_eq(diff[0][1], 4.0f), "float2x2 - col 0 failed");
    LUISA_ASSERT(float_eq(diff[1][0], 4.0f) && float_eq(diff[1][1], 4.0f), "float2x2 - col 1 failed");

    LUISA_INFO("test_matrix_addition_subtraction passed");
}

void test_make_functions() {
    // Test make_float2 variations
    float2 f2a = make_float2(1.0f);
    LUISA_ASSERT(f2a.x == 1.0f && f2a.y == 1.0f, "make_float2(scalar) failed");

    float2 f2b = make_float2(1.0f, 2.0f);
    LUISA_ASSERT(f2b.x == 1.0f && f2b.y == 2.0f, "make_float2(x,y) failed");

    float2 f2c = make_float2(float3(1.0f, 2.0f, 3.0f));
    LUISA_ASSERT(f2c.x == 1.0f && f2c.y == 2.0f, "make_float2(float3) failed");

    float2 f2d = make_float2(float4(1.0f, 2.0f, 3.0f, 4.0f));
    LUISA_ASSERT(f2d.x == 1.0f && f2d.y == 2.0f, "make_float2(float4) failed");

    // Test make_float3 variations
    float3 f3a = make_float3(1.0f);
    LUISA_ASSERT(f3a.x == 1.0f && f3a.y == 1.0f && f3a.z == 1.0f, "make_float3(scalar) failed");

    float3 f3b = make_float3(1.0f, 2.0f, 3.0f);
    LUISA_ASSERT(f3b.x == 1.0f && f3b.y == 2.0f && f3b.z == 3.0f, "make_float3(x,y,z) failed");

    float3 f3c = make_float3(float2(1.0f, 2.0f), 3.0f);
    LUISA_ASSERT(f3c.x == 1.0f && f3c.y == 2.0f && f3c.z == 3.0f, "make_float3(float2,z) failed");

    float3 f3d = make_float3(1.0f, float2(2.0f, 3.0f));
    LUISA_ASSERT(f3d.x == 1.0f && f3d.y == 2.0f && f3d.z == 3.0f, "make_float3(x,float2) failed");

    float3 f3e = make_float3(float4(1.0f, 2.0f, 3.0f, 4.0f));
    LUISA_ASSERT(f3e.x == 1.0f && f3e.y == 2.0f && f3e.z == 3.0f, "make_float3(float4) failed");

    // Test make_float4 variations
    float4 f4a = make_float4(1.0f);
    LUISA_ASSERT(f4a.x == 1.0f && f4a.y == 1.0f && f4a.z == 1.0f && f4a.w == 1.0f, "make_float4(scalar) failed");

    float4 f4b = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    LUISA_ASSERT(f4b.x == 1.0f && f4b.y == 2.0f && f4b.z == 3.0f && f4b.w == 4.0f, "make_float4(x,y,z,w) failed");

    float4 f4c = make_float4(float2(1.0f, 2.0f), 3.0f, 4.0f);
    LUISA_ASSERT(f4c.x == 1.0f && f4c.y == 2.0f && f4c.z == 3.0f && f4c.w == 4.0f, "make_float4(float2,z,w) failed");

    float4 f4d = make_float4(float3(1.0f, 2.0f, 3.0f), 4.0f);
    LUISA_ASSERT(f4d.x == 1.0f && f4d.y == 2.0f && f4d.z == 3.0f && f4d.w == 4.0f, "make_float4(float3,w) failed");

    float4 f4e = make_float4(1.0f, float3(2.0f, 3.0f, 4.0f));
    LUISA_ASSERT(f4e.x == 1.0f && f4e.y == 2.0f && f4e.z == 3.0f && f4e.w == 4.0f, "make_float4(x,float3) failed");

    float4 f4f = make_float4(float2(1.0f, 2.0f), float2(3.0f, 4.0f));
    LUISA_ASSERT(f4f.x == 1.0f && f4f.y == 2.0f && f4f.z == 3.0f && f4f.w == 4.0f, "make_float4(float2,float2) failed");

    // Test make_float2x2
    float2x2 m2a = make_float2x2(2.0f);
    LUISA_ASSERT(m2a[0][0] == 2.0f && m2a[1][1] == 2.0f, "make_float2x2(scalar) diagonal failed");
    LUISA_ASSERT(m2a[0][1] == 0.0f && m2a[1][0] == 0.0f, "make_float2x2(scalar) off-diagonal failed");

    float2x2 m2b = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
    LUISA_ASSERT(m2b[0][0] == 1.0f && m2b[0][1] == 2.0f, "make_float2x2(elements) col 0 failed");
    LUISA_ASSERT(m2b[1][0] == 3.0f && m2b[1][1] == 4.0f, "make_float2x2(elements) col 1 failed");

    float2x2 m2c = make_float2x2(float2(1.0f, 2.0f), float2(3.0f, 4.0f));
    LUISA_ASSERT(m2c[0][0] == 1.0f && m2c[1][0] == 3.0f, "make_float2x2(cols) failed");

    // Test make_float3x3
    float3x3 m3a = make_float3x3(2.0f);
    LUISA_ASSERT(m3a[0][0] == 2.0f && m3a[1][1] == 2.0f && m3a[2][2] == 2.0f, "make_float3x3(scalar) diagonal failed");

    float3x3 m3b = make_float3x3(
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f);
    LUISA_ASSERT(m3b[0][0] == 1.0f && m3b[0][1] == 2.0f && m3b[0][2] == 3.0f, "make_float3x3(elements) col 0 failed");
    LUISA_ASSERT(m3b[1][0] == 4.0f && m3b[1][1] == 5.0f && m3b[1][2] == 6.0f, "make_float3x3(elements) col 1 failed");
    LUISA_ASSERT(m3b[2][0] == 7.0f && m3b[2][1] == 8.0f && m3b[2][2] == 9.0f, "make_float3x3(elements) col 2 failed");

    // Test make_float4x4
    float4x4 m4a = make_float4x4(2.0f);
    LUISA_ASSERT(m4a[0][0] == 2.0f && m4a[1][1] == 2.0f && m4a[2][2] == 2.0f && m4a[3][3] == 2.0f, "make_float4x4(scalar) diagonal failed");

    float4x4 m4b = make_float4x4(
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f);
    LUISA_ASSERT(m4b[0][0] == 1.0f && m4b[0][3] == 4.0f, "make_float4x4(elements) col 0 failed");
    LUISA_ASSERT(m4b[3][0] == 13.0f && m4b[3][3] == 16.0f, "make_float4x4(elements) col 3 failed");

    LUISA_INFO("test_make_functions passed");
}

void test_type_aliases() {
    // Test that all vector type aliases exist and work
    bool2 b2; bool3 b3; bool4 b4;
    short2 s2; short3 s3; short4 s4;
    ushort2 us2; ushort3 us3; ushort4 us4;
    byte2 by2; byte3 by3; byte4 by4;
    ubyte2 ub2; ubyte3 ub3; ubyte4 ub4;
    int2 i2; int3 i3; int4 i4;
    uint2 u2; uint3 u3; uint4 u4;
    slong2 l2; slong3 l3; slong4 l4;
    ulong2 ul2; ulong3 ul3; ulong4 ul4;
    half2 h2; half3 h3; half4 h4;
    float2 f2; float3 f3; float4 f4;
    double2 d2; double3 d3; double4 d4;

    // Test that matrix type aliases exist and work
    float2x2 f22; float3x3 f33; float4x4 f44;
    double2x2 d22; double3x3 d33; double4x4 d44;
    half2x2 h22; half3x3 h33; half4x4 h44;

    LUISA_INFO("test_type_aliases passed");
}

void test_double_matrix_operations() {
    // Test double matrix scalar multiplication
    double2x2 m(double2(1.0, 2.0), double2(3.0, 4.0));
    double2x2 scaled = m * 2.0;
    LUISA_ASSERT(double_eq(scaled[0][0], 2.0) && double_eq(scaled[0][1], 4.0), "double2x2 * scalar failed");
    LUISA_ASSERT(double_eq(scaled[1][0], 6.0) && double_eq(scaled[1][1], 8.0), "double2x2 * scalar col 1 failed");

    double2x2 scaled2 = 3.0 * m;
    LUISA_ASSERT(double_eq(scaled2[0][0], 3.0), "scalar * double2x2 failed");

    // Test double matrix-vector multiplication
    double2 v(1.0, 1.0);
    double2 r = m * v;
    LUISA_ASSERT(double_eq(r.x, 4.0) && double_eq(r.y, 6.0), "double2x2 * double2 failed");

    // Test double matrix-matrix multiplication
    double2x2 a(double2(1.0, 2.0), double2(3.0, 4.0));
    double2x2 b(double2(5.0, 6.0), double2(7.0, 8.0));
    double2x2 c = a * b;
    LUISA_ASSERT(double_eq(c[0][0], 23.0) && double_eq(c[0][1], 34.0), "double2x2 * double2x2 col 0 failed");
    LUISA_ASSERT(double_eq(c[1][0], 31.0) && double_eq(c[1][1], 46.0), "double2x2 * double2x2 col 1 failed");

    // Test double matrix addition
    double2x2 sum = a + b;
    LUISA_ASSERT(double_eq(sum[0][0], 6.0), "double2x2 + failed");

    // Test double matrix subtraction
    double2x2 diff = b - a;
    LUISA_ASSERT(double_eq(diff[0][0], 4.0), "double2x2 - failed");

    LUISA_INFO("test_double_matrix_operations passed");
}

int main() {
    test_vector_construction();
    test_vector_element_access();
    test_vector_unary_operators();
    test_vector_binary_operators();
    test_vector_logic_operators();
    test_vector_assignment_operators();
    test_bool_vector_functions();
    test_matrix_construction();
    test_matrix_element_access();
    test_matrix_scalar_operators();
    test_matrix_vector_multiplication();
    test_matrix_matrix_multiplication();
    test_matrix_addition_subtraction();
    test_make_functions();
    test_type_aliases();
    test_double_matrix_operations();

    LUISA_INFO("All basic_types tests passed!");
    return 0;
}
