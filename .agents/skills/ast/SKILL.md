---
name: ast
description: Manual AST construction API for building kernels and callables without DSL syntax sugar
---

# Manual AST Construction

## Overview

LuisaCompute provides two approaches to construct kernels and callables:

1. **DSL Approach**: Using `Kernel2D<>`, `Callable<>` with C++ lambda syntax sugar
2. **Manual AST Approach**: Using `FunctionBuilder` to construct raw AST nodes directly

The manual AST approach is useful for:
- Code generation and metaprogramming
- Building kernels programmatically
- Cases where DSL syntax is not suitable
- Lower-level control over the generated code

**Key Header**: `#include <luisa/ast/function_builder.h>`

## FunctionBuilder

`FunctionBuilder` is the core class for constructing kernel/callable functions programmatically.

### Static Factory Methods

```cpp
using FuncBuilder = luisa::compute::detail::FunctionBuilder;

// Define a kernel
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    // ... build AST
});

// Define a callable
auto callable = FuncBuilder::define_callable([&]() {
    auto &cur = *FuncBuilder::current();
    // ... build AST
});

// Define a raster stage
auto raster = FuncBuilder::define_raster_stage([&]() {
    auto &cur = *FuncBuilder::current();
    // ... build AST
});
```

### Built-in Variables

```cpp
// Get current builder
auto &cur = *FuncBuilder::current();

// Dispatch IDs
auto dispatch_id = cur.dispatch_id();      // uint3 dispatch_id
auto dispatch_size = cur.dispatch_size();  // uint3 dispatch_size

// Thread/Block IDs
auto thread_id = cur.thread_id();          // uint3 thread_id
auto block_id = cur.block_id();            // uint3 block_id
auto kernel_id = cur.kernel_id();          // uint3 kernel_id

// Warp operations
auto lane_id = cur.warp_lane_id();         // uint lane_id
auto lane_count = cur.warp_lane_count();   // uint lane_count
```

### Configuration

```cpp
// Set block size for kernels
cur.set_block_size(uint3(16, 16, 1));

// Set function name
cur.set_name("my_kernel");

// Mark variable usage (required for references)
cur.mark_variable_usage(var_uid, Usage::READ_WRITE);
```

## Variables

### Variable Creation

```cpp
// Function arguments (input values)
auto arg = cur.argument(Type::of<float3>());

// References (inout parameters, read-write)
auto ref = cur.reference(Type::of<uint2>());
cur.mark_variable_usage(ref->variable().uid(), Usage::READ_WRITE);

// Local variables
auto local_var = cur.local(Type::of<float>());

// Shared/threadgroup memory
auto shared_mem = cur.shared(Type::of<float>(), count);

// Buffers
auto buffer = cur.buffer(Type::of<float>());

// Textures
auto tex2d = cur.texture(Type::of<Image<float>>());
auto tex3d = cur.texture(Type::of<Image3D<float>>());

// Acceleration structure
auto accel = cur.accel();

// Bindless array
auto bindless = cur.bindless_array();
```

### Bindings

```cpp
// Buffer binding
cur.buffer_binding(buffer_var, handle, offset, size);

// Texture binding
cur.texture_binding(texture_var, handle, level);

// Bindless array binding
cur.bindless_array_binding(bindless_var, handle);

// Acceleration structure binding
cur.accel_binding(accel_var, handle);
```

## Expressions

### Literals

```cpp
// Create literal values
auto lit_float = cur.literal(Type::of<float>(), 1.0f);
auto lit_int = cur.literal(Type::of<int>(), 42);
auto lit_uint = cur.literal(Type::of<uint>(), 0u);
auto lit_bool = cur.literal(Type::of<bool>(), true);
```

### Binary Operations

```cpp
// Binary operations require type, op, lhs, rhs
auto sum = cur.binary(Type::of<float>(), BinaryOp::ADD, a, b);
auto diff = cur.binary(Type::of<float>(), BinaryOp::SUB, a, b);
auto prod = cur.binary(Type::of<float>(), BinaryOp::MUL, a, b);
auto quot = cur.binary(Type::of<float>(), BinaryOp::DIV, a, b);
auto rem = cur.binary(Type::of<int>(), BinaryOp::MOD, a, b);

// Bitwise operations
auto band = cur.binary(Type::of<uint>(), BinaryOp::BIT_AND, a, b);
auto bor = cur.binary(Type::of<uint>(), BinaryOp::BIT_OR, a, b);
auto bxor = cur.binary(Type::of<uint>(), BinaryOp::BIT_XOR, a, b);
auto shl = cur.binary(Type::of<uint>(), BinaryOp::SHL, a, b);
auto shr = cur.binary(Type::of<uint>(), BinaryOp::SHR, a, b);

// Logical operations
auto logical_and = cur.binary(Type::of<bool>(), BinaryOp::AND, a, b);
auto logical_or = cur.binary(Type::of<bool>(), BinaryOp::OR, a, b);

// Comparison operations
auto eq = cur.binary(Type::of<bool>(), BinaryOp::EQ, a, b);
auto ne = cur.binary(Type::of<bool>(), BinaryOp::NE, a, b);
auto lt = cur.binary(Type::of<bool>(), BinaryOp::LT, a, b);
auto le = cur.binary(Type::of<bool>(), BinaryOp::LE, a, b);
auto gt = cur.binary(Type::of<bool>(), BinaryOp::GT, a, b);
auto ge = cur.binary(Type::of<bool>(), BinaryOp::GE, a, b);
```

### Unary Operations

```cpp
auto neg = cur.unary(Type::of<float>(), UnaryOp::MINUS, value);
auto pos = cur.unary(Type::of<float>(), UnaryOp::PLUS, value);
auto logical_not = cur.unary(Type::of<bool>(), UnaryOp::NOT, value);
auto bit_not = cur.unary(Type::of<uint>(), UnaryOp::BIT_NOT, value);
```

### Swizzle Operations

Swizzle encoding: component indices packed in nibbles (4 bits each), starting from lowest bits.

```cpp
// Swizzle .xy from uint3 to uint2
// Encoding: (0ull) | (1ull << 4ull) = components 0 and 1
uint64_t swizzle_xy = (0ull) | (1ull << 4ull);
auto coord = cur.swizzle(Type::of<uint2>(), coord_uint3, 2, swizzle_xy);

// Swizzle .xyzw (all components) - 0x3210u
auto vec4 = cur.swizzle(Type::of<float4>(), vec, 4, 0x3210u);

// Swizzle .x only
auto x_only = cur.swizzle(Type::of<float>(), vec, 1, 0ull);

// Swizzle .y only  
auto y_only = cur.swizzle(Type::of<float>(), vec, 1, 1ull << 4ull);
```

### Function Calls

```cpp
// Built-in call operations
auto value = cur.call(Type::of<float4>(), CallOp::MAKE_FLOAT4, 
                      {r, g, b, a});

cur.call(CallOp::TEXTURE_WRITE, {texture, coord, color});

auto color = cur.call(Type::of<float4>(), CallOp::TEXTURE_READ, 
                      {texture, coord});

// Call a custom callable
auto callable_result = cur.call(Type::of<float>(), Function(callable.get()), 
                                {arg1, arg2, arg3});

// Buffer operations
auto val = cur.call(Type::of<float>(), CallOp::BUFFER_READ, {buffer, index});
cur.call(CallOp::BUFFER_WRITE, {buffer, index, value});

// Atomic operations
auto old = cur.call(Type::of<uint>(), CallOp::ATOMIC_EXCHANGE, 
                    {atomic_buffer, index, new_value});
```

### Type Casting

```cpp
auto casted = cur.cast(Type::of<float>(), CastOp::STATIC, int_value);
```

### Array/Buffer Access

```cpp
auto elem = cur.access(Type::of<float>(), buffer_expr, index_expr);
```

### Struct Member Access

```cpp
auto member = cur.member(Type::of<float>(), struct_expr, member_index);
```

### Vector Construction

```cpp
auto vec2 = cur.make_vector(Type::of<float2>(), {x, y});
auto vec3 = cur.make_vector(Type::of<float3>(), {x, y, z});
auto vec4 = cur.make_vector(Type::of<float4>(), {x, y, z, w});
```

## Statements

### Assignment

```cpp
cur.assign(lhs_expr, rhs_expr);
```

### Control Flow

```cpp
// Break
cur.break_();

// Continue
cur.continue_();

// Return
cur.return_(value_expr);  // with value
cur.return_();            // void return

// If statement
auto if_stmt = cur.if_(condition_expr);
// Build true branch
auto &true_branch = *if_stmt->true_branch();
// ... add statements to true_branch
// Build false branch  
auto &false_branch = *if_stmt->false_branch();
// ... add statements to false_branch

// Loop statement
auto loop_stmt = cur.loop_();
auto &body = *loop_stmt->body();
// ... add statements to body

// For loop
auto for_stmt = cur.for_(var_expr, cond_expr, step_expr);
auto &for_body = *for_stmt->body();
// ... add statements to for_body

// Switch statement
auto switch_stmt = cur.switch_(expr);
auto &switch_body = *switch_stmt->body();
// ... add case statements
```

### Expression Statement

For calls that don't produce values (side effects):

```cpp
cur.call(CallOp::TEXTURE_WRITE, {tex, coord, color});
// This is automatically wrapped in ExprStmt
```

### Print Statement

```cpp
cur.print_("value = {}", {value_expr});
```

### Ray Query

```cpp
auto ray_query_stmt = cur.ray_query_(query_expr);
auto &triangle_branch = *ray_query_stmt->on_triangle_candidate();
auto &procedural_branch = *ray_query_stmt->on_procedural_candidate();
```

## Type System

### Getting Types

```cpp
// From C++ types
auto float_type = Type::of<float>();
auto int_type = Type::of<int>();
auto uint_type = Type::of<uint>();
auto bool_type = Type::of<bool>();

// Vector types
auto float2_type = Type::of<float2>();
auto float3_type = Type::of<float3>();
auto float4_type = Type::of<float4>();
auto uint2_type = Type::of<uint2>();
auto uint3_type = Type::of<uint3>();
auto int2_type = Type::of<int2>();

// Matrix types
auto float3x3_type = Type::of<float3x3>();
auto float4x4_type = Type::of<float4x4>();

// Resource types
auto buffer_type = Type::of<Buffer<float>>();
auto image2d_type = Type::of<Image<float>>();
auto image3d_type = Type::of<Image3D<float>>();
auto accel_type = Type::of<Accel>();
auto bindless_type = Type::of<BindlessArray>();
```

### Constructing Types

```cpp
// Vector type
auto vec2_type = Type::vector(Type::of<float>(), 2);

// Matrix type
auto mat4_type = Type::matrix(4);

// Array type
auto array_type = Type::array(Type::of<float>(), 100);

// Struct type
auto struct_type = Type::structure({Type::of<float>(), Type::of<int>(), Type::of<uint>()});

// Buffer type
auto buffer_type = Type::buffer(Type::of<float>());

// Texture type
auto tex2d_type = Type::texture(Type::of<float>(), 2);  // 2D texture
auto tex3d_type = Type::texture(Type::of<float>(), 3);  // 3D texture

// From string description
auto type = Type::from("vector<float,4>");
auto array_type = Type::from("array<float,100>");
auto buffer_type = Type::from("buffer<float>");
```

## Operators

### BinaryOp (from op.h)

```cpp
enum struct BinaryOp : uint32_t {
    ADD, SUB, MUL, DIV, MOD,           // Arithmetic
    BIT_AND, BIT_OR, BIT_XOR,          // Bitwise
    SHL, SHR,                          // Shift
    AND, OR,                           // Logical
    EQ, NE, LT, LE, GT, GE             // Comparison
};
```

### UnaryOp (from op.h)

```cpp
enum struct UnaryOp : uint32_t {
    PLUS, MINUS, NOT, BIT_NOT
};
```

### CallOp (from op.h)

Common call operations:

```cpp
// Vector construction
CallOp::MAKE_FLOAT2, CallOp::MAKE_FLOAT3, CallOp::MAKE_FLOAT4
CallOp::MAKE_INT2, CallOp::MAKE_INT3, CallOp::MAKE_INT4
CallOp::MAKE_UINT2, CallOp::MAKE_UINT3, CallOp::MAKE_UINT4
CallOp::MAKE_BOOL2, CallOp::MAKE_BOOL3, CallOp::MAKE_BOOL4

// Buffer operations
CallOp::BUFFER_READ, CallOp::BUFFER_WRITE
CallOp::BUFFER_SIZE

// Texture operations
CallOp::TEXTURE_READ, CallOp::TEXTURE_WRITE
CallOp::TEXTURE_SAMPLE, CallOp::TEXTURE_SAMPLE_LEVEL
CallOp::TEXTURE_SAMPLE_GRAD
CallOp::TEXTURE_SIZE, CallOp::TEXTURE_SIZE_LEVEL

// Atomic operations
CallOp::ATOMIC_EXCHANGE, CallOp::ATOMIC_COMPARE_EXCHANGE
CallOp::ATOMIC_FETCH_ADD, CallOp::ATOMIC_FETCH_SUB
CallOp::ATOMIC_FETCH_AND, CallOp::ATOMIC_FETCH_OR
CallOp::ATOMIC_FETCH_XOR, CallOp::ATOMIC_FETCH_MIN
CallOp::ATOMIC_FETCH_MAX

// Ray tracing
CallOp::RAY_TRACING_TRACE_CLOSEST, CallOp::RAY_TRACING_TRACE_ANY
CallOp::RAY_TRACING_QUERY_ALL, CallOp::RAY_TRACING_QUERY_ANY
CallOp::RAY_TRACING_INSTANCE_MOTION_MATRIX
CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX
CallOp::RAY_TRACING_INSTANCE_MOTION_SRT
CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT

// Math
CallOp::ABS, CallOp::MIN, CallOp::MAX
CallOp::CLAMP, CallOp::LERP, CallOp::SMOOTHSTEP
CallOp::SATURATE
CallOp::FMA

// Trigonometry
CallOp::SIN, CallOp::COS, CallOp::TAN
CallOp::ASIN, CallOp::ACOS, CallOp::ATAN, CallOp::ATAN2
CallOp::SINH, CallOp::COSH, CallOp::TANH

// Exponential/Logarithmic
CallOp::EXP, CallOp::EXP2, CallOp::EXP10
CallOp::LOG, CallOp::LOG2, CallOp::LOG10
CallOp::POW, CallOp::SQRT, CallOp::RSQRT

// Rounding
CallOp::CEIL, CallOp::FLOOR, CallOp::FRACT
CallOp::TRUNC, CallOp::ROUND

// Vector operations
CallOp::DOT, CallOp::CROSS
CallOp::LENGTH, CallOp::LENGTH_SQUARED
CallOp::NORMALIZE, CallOp::FACEFORWARD
CallOp::REFLECT, CallOp::REFRACT

// Matrix operations
CallOp::MATRIX_DETERMINANT, CallOp::MATRIX_INVERSE
CallOp::MATRIX_TRANSPOSE
CallOp::MATRIX_COMPOSE, CallOp::MATRIX_DECOMPOSE
CallOp::MATRIX_MULTIPLY

// Warp operations
CallOp::WARP_IS_FIRST_ACTIVE_LANE
CallOp::WARP_FIRST_ACTIVE_LANE
CallOp::WARP_ACTIVE_ALL_EQUAL
CallOp::WARP_ACTIVE_BIT_AND, CallOp::WARP_ACTIVE_BIT_OR, CallOp::WARP_ACTIVE_BIT_XOR
CallOp::WARP_ACTIVE_COUNT_BITS, CallOp::WARP_ACTIVE_MAX, CallOp::WARP_ACTIVE_MIN
CallOp::WARP_ACTIVE_PRODUCT, CallOp::WARP_ACTIVE_SUM
CallOp::WARP_ACTIVE_ALL, CallOp::WARP_ACTIVE_ANY
CallOp::WARP_ACTIVE_BIT_COUNT
CallOp::WARP_PREFIX_SUM, CallOp::WARP_PREFIX_PRODUCT

// Synchronization
CallOp::SYNCHRONIZE_BLOCK

// Subgroup operations
CallOp::SUBGROUP_BARRIER
```

## Usage Flags

```cpp
enum struct Usage : uint32_t {
    NONE = 0u,
    READ = 0x01u,
    WRITE = 0x02u,
    READ_WRITE = READ | WRITE
};
```

**Important**: References (inout parameters) require explicit usage marking:

```cpp
auto ref = cur.reference(Type::of<uint2>());
cur.mark_variable_usage(ref->variable().uid(), Usage::READ_WRITE);
```

## Complete Examples

### Example 1: Simple Kernel

```cpp
#include <luisa/ast/function_builder.h>
#include <luisa/ast/type.h>
#include <luisa/ast/op.h>
#include <luisa/ast/usage.h>

using namespace luisa::compute;
using FuncBuilder = luisa::compute::detail::FunctionBuilder;

// Define a simple kernel that clears an image to a color
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    
    // Set block size
    cur.set_block_size(uint3(16, 16, 1));
    
    // Get dispatch ID (pixel coordinate)
    auto dispatch = cur.dispatch_id();
    
    // Get image argument
    auto img = cur.texture(Type::of<Image<float>>());
    auto color = cur.argument(Type::of<float4>());
    
    // Write color to texture
    cur.call(CallOp::TEXTURE_WRITE, {img, dispatch, color});
});
```

### Example 2: Callable with Reference

```cpp
// Define a callable that modifies a coordinate and writes to a texture
auto callable = FuncBuilder::define_callable([&]() {
    auto &cur = *FuncBuilder::current();
    
    // Texture parameter
    auto tex = cur.texture(Type::of<Image<float>>());
    
    // Reference parameter (read-write)
    auto coord_ref = cur.reference(Type::of<uint2>());
    cur.mark_variable_usage(coord_ref->variable().uid(), Usage::READ_WRITE);
    
    // Color argument
    auto color = cur.argument(Type::of<float3>());
    
    // Create float4 from float3 + alpha
    auto alpha = cur.literal(Type::of<float>(), 1.0f);
    auto value = cur.call(Type::of<float4>(), CallOp::MAKE_FLOAT4, {color, alpha});
    
    // Write to texture
    cur.call(CallOp::TEXTURE_WRITE, {tex, coord_ref, value});
});
```

### Example 3: Kernel Calling Callable

```cpp
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    
    cur.set_block_size(uint3(16, 16, 1));
    
    // Kernel arguments
    auto img = cur.texture(Type::of<Image<float>>());
    auto color = cur.argument(Type::of<float3>());
    
    // Get dispatch ID and convert to uint2 coordinate
    auto coord_uint3 = cur.dispatch_id();
    auto coord = cur.local(Type::of<uint2>());
    
    // Swizzle .xy from uint3
    uint64_t swizzle_xy = (0ull) | (1ull << 4ull);
    cur.assign(coord, cur.swizzle(Type::of<uint2>(), coord_uint3, 2, swizzle_xy));
    
    // Call the callable
    cur.call(Function(callable.get()), {img, coord, color});
});
```

### Example 4: Swizzle Operations

```cpp
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    
    auto input = cur.argument(Type::of<float4>());
    auto output = cur.reference(Type::of<float4>());
    cur.mark_variable_usage(output->variable().uid(), Usage::READ_WRITE);
    
    // Extract .xyz (components 0,1,2)
    uint64_t swizzle_xyz = (0ull) | (1ull << 4ull) | (2ull << 8ull);
    auto xyz = cur.swizzle(Type::of<float3>(), input, 3, swizzle_xyz);
    
    // Extract .w (component 3)
    auto w = cur.swizzle(Type::of<float>(), input, 1, 3ull << 4ull);
    
    // Create new float4 with swizzled components
    auto result = cur.call(Type::of<float4>(), CallOp::MAKE_FLOAT4, {xyz, w});
    cur.assign(output, result);
});
```

### Example 5: Buffer Operations

```cpp
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    
    cur.set_block_size(uint3(256, 1, 1));
    
    auto input_buf = cur.buffer(Type::of<float>());
    auto output_buf = cur.buffer(Type::of<float>());
    
    // Get thread index
    auto tid = cur.thread_id();
    auto idx = cur.swizzle(Type::of<uint>(), tid, 1, 0ull);
    
    // Read from input buffer
    auto value = cur.call(Type::of<float>(), CallOp::BUFFER_READ, {input_buf, idx});
    
    // Compute: value = value * 2.0 + 1.0
    auto two = cur.literal(Type::of<float>(), 2.0f);
    auto one = cur.literal(Type::of<float>(), 1.0f);
    auto scaled = cur.binary(Type::of<float>(), BinaryOp::MUL, value, two);
    auto result = cur.binary(Type::of<float>(), BinaryOp::ADD, scaled, one);
    
    // Write to output buffer
    cur.call(CallOp::BUFFER_WRITE, {output_buf, idx, result});
});
```

## Important Notes

1. **Always use `Type::of<T>()`** for type specifications - the AST requires explicit type information

2. **Mark reference usage** - When using `reference()`, call `mark_variable_usage(uid, Usage::READ_WRITE)`

3. **Swizzle encoding** - Component indices are packed in nibbles (4 bits), starting from lowest bits:
   - `.x` = `0ull`
   - `.y` = `1ull << 4ull`
   - `.xy` = `(0ull) | (1ull << 4ull)`
   - `.xyzw` = `0x3210u`

4. **Builder access** - Use `FunctionBuilder::current()` to get the current builder within define callbacks

5. **Statement ownership** - Statements added to scopes are owned by the FunctionBuilder; don't manually delete them

6. **Block size** - Call `set_block_size()` for compute kernels (typically uint3(16, 16, 1) for 2D work)
