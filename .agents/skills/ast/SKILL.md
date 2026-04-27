---
name: ast
description: Manual AST construction API for building kernels and callables without DSL syntax sugar
---

# Manual AST Construction

## Overview

Two approaches to construct kernels/callables:
- **DSL**: `Kernel2D<>`, `Callable<>` with C++ lambdas
- **Manual AST**: `FunctionBuilder` for direct AST construction

Use manual AST for: code generation, metaprogramming, programmatic kernel building, lower-level control.

**Header**: `#include <luisa/ast/function_builder.h>`

## FunctionBuilder

```cpp
using FuncBuilder = luisa::compute::detail::FunctionBuilder;

// Define functions
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    // ... build AST
});

auto callable = FuncBuilder::define_callable([&]() { ... });
auto raster = FuncBuilder::define_raster_stage([&]() { ... });
```

### Built-in Variables
```cpp
auto &cur = *FuncBuilder::current();

cur.dispatch_id();      // uint3 dispatch_id
cur.dispatch_size();    // uint3 dispatch_size
cur.thread_id();        // uint3 thread_id
cur.block_id();         // uint3 block_id
cur.kernel_id();        // uint3 kernel_id
cur.warp_lane_id();     // uint lane_id
cur.warp_lane_count();  // uint lane_count
```

### Configuration
```cpp
cur.set_block_size(uint3(16, 16, 1));        // Kernel block size
cur.set_name("my_kernel");                   // Function name
cur.mark_variable_usage(var_uid, Usage::READ_WRITE);  // Reference usage
```

## Variables

```cpp
// Arguments
cur.argument(Type::of<float3>());            // Input value
auto ref = cur.reference(Type::of<uint2>()); // Inout parameter
cur.mark_variable_usage(ref->variable().uid(), Usage::READ_WRITE);

// Memory
cur.local(Type::of<float>());                // Local variable
cur.shared(Type::of<float>(), count);        // Shared/threadgroup memory

// Resources
cur.buffer(Type::of<float>());               // Buffer
cur.texture(Type::of<Image<float>>());       // 2D texture
cur.texture(Type::of<Image3D<float>>());     // 3D texture
cur.accel();                                 // Acceleration structure
cur.bindless_array();                        // Bindless array
```

### Bindings
```cpp
cur.buffer_binding(buffer_var, handle, offset, size);
cur.texture_binding(texture_var, handle, level);
cur.bindless_array_binding(bindless_var, handle);
cur.accel_binding(accel_var, handle);
```

## Expressions

### Literals
```cpp
cur.literal(Type::of<float>(), 1.0f);
cur.literal(Type::of<int>(), 42);
cur.literal(Type::of<uint>(), 0u);
cur.literal(Type::of<bool>(), true);
```

### Binary Operations
```cpp
cur.binary(Type::of<float>(), BinaryOp::ADD, a, b);   // +
cur.binary(Type::of<float>(), BinaryOp::SUB, a, b);   // -
cur.binary(Type::of<float>(), BinaryOp::MUL, a, b);   // *
cur.binary(Type::of<float>(), BinaryOp::DIV, a, b);   // /
cur.binary(Type::of<int>(), BinaryOp::MOD, a, b);     // %
cur.binary(Type::of<uint>(), BinaryOp::BIT_AND, a, b);// &
cur.binary(Type::of<uint>(), BinaryOp::BIT_OR, a, b); // |
cur.binary(Type::of<uint>(), BinaryOp::BIT_XOR, a, b);// ^
cur.binary(Type::of<uint>(), BinaryOp::SHL, a, b);    // <<
cur.binary(Type::of<uint>(), BinaryOp::SHR, a, b);    // >>
cur.binary(Type::of<bool>(), BinaryOp::AND, a, b);    // &&
cur.binary(Type::of<bool>(), BinaryOp::OR, a, b);     // ||
cur.binary(Type::of<bool>(), BinaryOp::EQ, a, b);     // ==
cur.binary(Type::of<bool>(), BinaryOp::NE, a, b);     // !=
cur.binary(Type::of<bool>(), BinaryOp::LT, a, b);     // <
cur.binary(Type::of<bool>(), BinaryOp::LE, a, b);     // <=
cur.binary(Type::of<bool>(), BinaryOp::GT, a, b);     // >
cur.binary(Type::of<bool>(), BinaryOp::GE, a, b);     // >=
```

### Unary Operations
```cpp
cur.unary(Type::of<float>(), UnaryOp::MINUS, value);   // -
cur.unary(Type::of<float>(), UnaryOp::PLUS, value);    // +
cur.unary(Type::of<bool>(), UnaryOp::NOT, value);      // !
cur.unary(Type::of<uint>(), UnaryOp::BIT_NOT, value);  // ~
```

### Swizzle Operations
Component indices packed in nibbles (4 bits), lowest bits first:
```cpp
// .xy from uint3: (0) | (1 << 4)
uint64_t swizzle_xy = (0ull) | (1ull << 4ull);
auto coord = cur.swizzle(Type::of<uint2>(), coord_uint3, 2, swizzle_xy);

// .xyzw (all): 0x3210u
cur.swizzle(Type::of<float4>(), vec, 4, 0x3210u);

// .x only: 0ull
// .y only: 1ull << 4
// .z only: 2ull << 8
// .w only: 3ull << 12
```

### Function Calls
```cpp
// Built-in calls
cur.call(Type::of<float4>(), CallOp::MAKE_FLOAT4, {r, g, b, a});
cur.call(CallOp::TEXTURE_WRITE, {texture, coord, color});
cur.call(Type::of<float4>(), CallOp::TEXTURE_READ, {texture, coord});

// Buffer operations
cur.call(Type::of<float>(), CallOp::BUFFER_READ, {buffer, index});
cur.call(CallOp::BUFFER_WRITE, {buffer, index, value});

// Atomic operations
cur.call(Type::of<uint>(), CallOp::ATOMIC_EXCHANGE, {atomic_buffer, index, new_value});

// Custom callable
cur.call(Function(callable.get()), {arg1, arg2});
```

### Other Expressions
```cpp
cur.cast(Type::of<float>(), CastOp::STATIC, int_value);  // Type cast
cur.access(Type::of<float>(), buffer_expr, index_expr);  // Array/buffer access
cur.member(Type::of<float>(), struct_expr, member_index);// Struct member
cur.make_vector(Type::of<float4>(), {x, y, z, w});       // Vector construction
```

## Statements

```cpp
// Assignment
cur.assign(lhs_expr, rhs_expr);

// Control flow
cur.break_();
cur.continue_();
cur.return_(value_expr);    // With value
cur.return_();              // Void return

// If statement
auto if_stmt = cur.if_(condition_expr);
auto &true_branch = *if_stmt->true_branch();
auto &false_branch = *if_stmt->false_branch();

// Loop statement
auto loop_stmt = cur.loop_();
auto &body = *loop_stmt->body();

// For loop
auto for_stmt = cur.for_(var_expr, cond_expr, step_expr);
auto &for_body = *for_stmt->body();

// Switch statement
auto switch_stmt = cur.switch_(expr);
auto &switch_body = *switch_stmt->body();

// Ray query
auto ray_query_stmt = cur.ray_query_(query_expr);
auto &triangle_branch = *ray_query_stmt->on_triangle_candidate();
auto &procedural_branch = *ray_query_stmt->on_procedural_candidate();

// Print
cur.print_("value = {}", {value_expr});
```

## Type System

### Getting Types
```cpp
// Scalars
Type::of<float>(); Type::of<int>(); Type::of<uint>(); Type::of<bool>();

// Vectors
Type::of<float2>(); Type::of<float3>(); Type::of<float4>();
Type::of<int2>(); Type::of<int3>(); Type::of<int4>();
Type::of<uint2>(); Type::of<uint3>(); Type::of<uint4>();

// Matrices
Type::of<float3x3>(); Type::of<float4x4>();

// Resources
Type::of<Buffer<float>>();
Type::of<Image<float>>(); Type::of<Image3D<float>>();
Type::of<Accel>(); Type::of<BindlessArray>();
```

### Constructing Types
```cpp
Type::vector(Type::of<float>(), 2);                    // Vector
Type::matrix(4);                                        // Matrix
Type::array(Type::of<float>(), 100);                   // Array
Type::structure({Type::of<float>(), Type::of<int>()}); // Struct
Type::buffer(Type::of<float>());                       // Buffer
Type::texture(Type::of<float>(), 2);                   // 2D texture
Type::texture(Type::of<float>(), 3);                   // 3D texture
Type::from("vector<float,4>");                         // From string
```

## Operators

### BinaryOp
```cpp
ADD, SUB, MUL, DIV, MOD,      // Arithmetic
BIT_AND, BIT_OR, BIT_XOR,     // Bitwise
SHL, SHR,                     // Shift
AND, OR,                      // Logical
EQ, NE, LT, LE, GT, GE        // Comparison
```

### UnaryOp
```cpp
PLUS, MINUS, NOT, BIT_NOT
```

### Common CallOp
```cpp
// Vector construction
MAKE_FLOAT2/3/4, MAKE_INT2/3/4, MAKE_UINT2/3/4, MAKE_BOOL2/3/4

// Buffer/Texture
BUFFER_READ, BUFFER_WRITE, BUFFER_SIZE
TEXTURE_READ, TEXTURE_WRITE, TEXTURE_SAMPLE, TEXTURE_SAMPLE_LEVEL

// Atomic
ATOMIC_EXCHANGE, ATOMIC_COMPARE_EXCHANGE, ATOMIC_FETCH_ADD, ATOMIC_FETCH_SUB
ATOMIC_FETCH_AND, ATOMIC_FETCH_OR, ATOMIC_FETCH_XOR, ATOMIC_FETCH_MIN, ATOMIC_FETCH_MAX

// Ray tracing
RAY_TRACING_TRACE_CLOSEST, RAY_TRACING_TRACE_ANY, RAY_TRACING_QUERY_ALL, RAY_TRACING_QUERY_ANY

// Math
ABS, MIN, MAX, CLAMP, LERP, SMOOTHSTEP, SATURATE, FMA
SIN, COS, TAN, ASIN, ACOS, ATAN, ATAN2, SINH, COSH, TANH
EXP, EXP2, EXP10, LOG, LOG2, LOG10, POW, SQRT, RSQRT
CEIL, FLOOR, FRACT, TRUNC, ROUND

// Vector/Matrix
DOT, CROSS, LENGTH, LENGTH_SQUARED, NORMALIZE, FACEFORWARD, REFLECT, REFRACT
MATRIX_DETERMINANT, MATRIX_INVERSE, MATRIX_TRANSPOSE, MATRIX_COMPOSE, MATRIX_DECOMPOSE, MATRIX_MULTIPLY

// Warp
WARP_IS_FIRST_ACTIVE_LANE, WARP_FIRST_ACTIVE_LANE, WARP_ACTIVE_ALL_EQUAL
WARP_ACTIVE_BIT_AND/OR/XOR, WARP_ACTIVE_COUNT_BITS, WARP_ACTIVE_MAX/MIN
WARP_ACTIVE_PRODUCT/SUM, WARP_ACTIVE_ALL/ANY, WARP_ACTIVE_BIT_COUNT
WARP_PREFIX_SUM, WARP_PREFIX_PRODUCT

// Sync
SYNCHRONIZE_BLOCK, SUBGROUP_BARRIER
```

## Usage Flags
```cpp
enum struct Usage : uint32_t {
    NONE = 0u, READ = 0x01u, WRITE = 0x02u, READ_WRITE = READ | WRITE
};
```
**Required for references**: `cur.mark_variable_usage(ref->variable().uid(), Usage::READ_WRITE);`

## Examples

### Simple Kernel
```cpp
#include <luisa/ast/function_builder.h>
#include <luisa/ast/type.h>
#include <luisa/ast/op.h>
#include <luisa/ast/usage.h>

using namespace luisa::compute;
using FuncBuilder = luisa::compute::detail::FunctionBuilder;

auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    cur.set_block_size(uint3(16, 16, 1));
    
    auto dispatch = cur.dispatch_id();
    auto img = cur.texture(Type::of<Image<float>>());
    auto color = cur.argument(Type::of<float4>());
    
    cur.call(CallOp::TEXTURE_WRITE, {img, dispatch, color});
});
```

### Callable with Reference
```cpp
auto callable = FuncBuilder::define_callable([&]() {
    auto &cur = *FuncBuilder::current();
    
    auto tex = cur.texture(Type::of<Image<float>>());
    auto coord_ref = cur.reference(Type::of<uint2>());
    cur.mark_variable_usage(coord_ref->variable().uid(), Usage::READ_WRITE);
    auto color = cur.argument(Type::of<float3>());
    
    auto alpha = cur.literal(Type::of<float>(), 1.0f);
    auto value = cur.call(Type::of<float4>(), CallOp::MAKE_FLOAT4, {color, alpha});
    cur.call(CallOp::TEXTURE_WRITE, {tex, coord_ref, value});
});
```

### Kernel Calling Callable
```cpp
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    cur.set_block_size(uint3(16, 16, 1));
    
    auto img = cur.texture(Type::of<Image<float>>());
    auto color = cur.argument(Type::of<float3>());
    
    auto coord_uint3 = cur.dispatch_id();
    auto coord = cur.local(Type::of<uint2>());
    uint64_t swizzle_xy = (0ull) | (1ull << 4ull);
    cur.assign(coord, cur.swizzle(Type::of<uint2>(), coord_uint3, 2, swizzle_xy));
    
    cur.call(Function(callable.get()), {img, coord, color});
});
```

### Swizzle Operations
```cpp
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    
    auto input = cur.argument(Type::of<float4>());
    auto output = cur.reference(Type::of<float4>());
    cur.mark_variable_usage(output->variable().uid(), Usage::READ_WRITE);
    
    uint64_t swizzle_xyz = (0ull) | (1ull << 4ull) | (2ull << 8ull);
    auto xyz = cur.swizzle(Type::of<float3>(), input, 3, swizzle_xyz);
    auto w = cur.swizzle(Type::of<float>(), input, 1, 3ull << 4ull);
    
    cur.assign(output, cur.call(Type::of<float4>(), CallOp::MAKE_FLOAT4, {xyz, w}));
});
```

### Buffer Operations
```cpp
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    cur.set_block_size(uint3(256, 1, 1));
    
    auto input_buf = cur.buffer(Type::of<float>());
    auto output_buf = cur.buffer(Type::of<float>());
    
    auto tid = cur.thread_id();
    auto idx = cur.swizzle(Type::of<uint>(), tid, 1, 0ull);
    
    auto value = cur.call(Type::of<float>(), CallOp::BUFFER_READ, {input_buf, idx});
    auto two = cur.literal(Type::of<float>(), 2.0f);
    auto one = cur.literal(Type::of<float>(), 1.0f);
    auto scaled = cur.binary(Type::of<float>(), BinaryOp::MUL, value, two);
    auto result = cur.binary(Type::of<float>(), BinaryOp::ADD, scaled, one);
    
    cur.call(CallOp::BUFFER_WRITE, {output_buf, idx, result});
});
```

## Important Notes

1. **Always use `Type::of<T>()`** for explicit type specifications
2. **Mark reference usage** with `mark_variable_usage(uid, Usage::READ_WRITE)`
3. **Swizzle encoding**: Component indices in nibbles (4 bits), lowest bits first
4. **Builder access**: Use `FunctionBuilder::current()` within define callbacks
5. **Statement ownership**: Owned by FunctionBuilder; don't manually delete
6. **Block size**: Call `set_block_size()` for compute kernels (typically uint3(16, 16, 1) for 2D work)
