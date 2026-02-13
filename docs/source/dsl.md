# Domain Specific Language (DSL)

LuisaCompute features an embedded Domain Specific Language (DSL) that allows you to write GPU kernels directly in modern C++. The DSL uses C++ metaprogramming techniques to trace your kernel code and build an Abstract Syntax Tree (AST) that is then compiled for different backends.

## Overview

The DSL provides:
- **Type-safe device variables** via `Var<T>`
- **Vector and matrix types** for graphics computing
- **Control flow** statements executed on the GPU
- **Built-in functions** for math, texture sampling, and ray tracing
- **Automatic differentiation** for gradient computation

## Basic Types

### Scalar Types

The following scalar types are supported in the DSL:

| C++ Type | DSL Alias | Description |
|----------|-----------|-------------|
| `bool` | `Bool` | Boolean value |
| `int` / `int32_t` | `Int` | 32-bit signed integer |
| `uint` / `uint32_t` | `UInt` | 32-bit unsigned integer |
| `float` | `Float` | 32-bit floating point |
| `short` | `Short` | 16-bit signed integer |
| `ushort` | `UShort` | 16-bit unsigned integer |
| `slong` / `int64_t` | `SLong` | 64-bit signed integer |
| `ulong` / `uint64_t` | `ULong` | 64-bit unsigned integer |
| `half` | `Half` | 16-bit floating point (IEEE 754) |

```cpp
Float x = 1.0f;           // Declare a float variable
Int count = 100;          // Declare an integer
Bool flag = true;         // Declare a boolean
```

### Vector Types

Vector types follow the naming convention `[Type][Dimension]`:

```cpp
// 2D vectors
Float2 pos = make_float2(1.0f, 2.0f);
Int2 coord = make_int2(100, 200);

// 3D vectors (aligned to 16 bytes)
Float3 color = make_float3(1.0f, 0.5f, 0.0f);
Float3 origin = make_float3(0.0f);

// 4D vectors
Float4 rgba = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
```

**Construction patterns:**

```cpp
// From scalar (broadcast)
Float2 v2 = make_float2(1.0f);        // (1.0f, 1.0f)
Float3 v3 = make_float3(0.5f);        // (0.5f, 0.5f, 0.5f)

// From components
Float2 a = make_float2(1.0f, 2.0f);
Float3 b = make_float3(a, 3.0f);      // (1.0f, 2.0f, 3.0f)

// From another vector (conversion)
Float3 f3 = make_float3(make_int3(1, 2, 3));  // (1.0f, 2.0f, 3.0f)
```

### Matrix Types

Column-major matrix types for 3D transformations:

```cpp
// Create identity matrices
Float2x2 m2 = make_float2x2(1.0f);
Float3x3 m3 = make_float3x3(1.0f);
Float4x4 m4 = make_float4x4(1.0f);

// From column vectors
Float4x4 transform = make_float4x4(
    make_float4(1, 0, 0, 0),  // First column
    make_float4(0, 1, 0, 0),  // Second column
    make_float4(0, 0, 1, 0),  // Third column
    make_float4(tx, ty, tz, 1) // Fourth column (translation)
);

// Matrix-vector multiplication
Float4 result = transform * make_float4(pos, 1.0f);
```

### Swizzling

Access vector components using GLSL-style swizzles:

```cpp
Float4 color = make_float4(1.0f, 0.5f, 0.25f, 1.0f);

// Extract components
Float3 rgb = color.xyz();     // (1.0, 0.5, 0.25)
Float red = color.x();        // 1.0
Float alpha = color.w();      // 1.0 (or use .a())

// Reordering
Float3 bgr = color.zyx();     // (0.25, 0.5, 1.0)
Float2 rg = color.xy();       // (1.0, 0.5)

// Replication
Float4 all_red = color.xxxx(); // (1.0, 1.0, 1.0, 1.0)
Float3 all_green = color.yyy(); // (0.5, 0.5, 0.5)

// Mixed swizzles
Float4 rgba = color.xyzw();   // Identity
Float2 ba = color.wz();       // (1.0, 0.25)
```

**Available swizzle components:**
- For vectors: `x`, `y`, `z`, `w` (position) and `r`, `g`, `b`, `a` (color)
- Any combination from 1 to 4 components is allowed

### Array Variables

Fixed-size arrays in the DSL:

```cpp
// Array of 10 floats
ArrayVar<float, 10> arr;

// Initialize with values
Float value = arr[0];         // Read
arr[1] = 3.14f;               // Write

// Array of vectors
ArrayVar<float3, 100> positions;
Float3 pos = positions[index];

// Convenience aliases
template<size_t N>
using ArrayFloat = ArrayVar<float, N>;
ArrayFloat<10> arr;  // Same as ArrayVar<float, 10>
```

## Variables and References

### Var<T>

`Var<T>` is the core type for representing device-side data:

```cpp
// Default initialization (zero for numeric types)
Float x;                      // x = 0.0f
Float3 v;                     // v = (0, 0, 0)

// Construction from value
Float y = 1.0f;
Float2 pos = make_float2(1.0f, 2.0f);

// Copy semantics (creates new variable, copies value)
Float z = y;                  // z is a new variable with y's value

// Assignment (records assignment in AST)
z = 2.0f;                     // Records: z = 2.0f
```

### Expr<T>

`Expr<T>` represents a computed expression without creating a new variable:

```cpp
Float a = 1.0f;
Float b = 2.0f;

// This creates a new variable and assigns the sum
Float c = a + b;

// Expr<T> can hold expressions without variable creation
Expr<Float> sum = a + b;      // No new variable, just the expression

// Useful for passing to functions
auto result = some_function(Expr{a + b});
```

### def<T>() Helper

Convert C++ values to DSL expressions:

```cpp
// Convert literal
auto x = def(3.14f);          // Float(3.14f)
auto i = def(42);             // Int(42)

// Convert host variable
float host_value = 1.5f;
auto y = def(host_value);     // Float(1.5f)

// Explicit type
auto z = def<float>(2.0);     // Float(2.0f)

// User-defined literals
using namespace luisa::compute::dsl_literals;
auto a = 3.14_float;          // Float(3.14f)
auto b = 100_uint;            // UInt(100)
auto c = make_float2(1.0_float, 2.0_float);
```

## Control Flow

DSL control flow uses macros prefixed with `$`. These are executed on the GPU at runtime.

### Conditional Statements

```cpp
// Simple if
$if (x > 0.0f) {
    // GPU-side branch
    y = sqrt(x);
};

// If-else
$if (x < 0.0f) {
    result = -x;
} $else {
    result = x;
};

// If-elif-else chain
$if (grade >= 90) {
    letter = 'A';
} $elif (grade >= 80) {
    letter = 'B';
} $elif (grade >= 70) {
    letter = 'C';
} $else {
    letter = 'F';
};
```

### Loops

```cpp
// While loop
Var<int> i = 0;
$while (i < 100) {
    // Do something
    i = i + 1;  // Don't forget to update the counter!
};

// Range-based for loop (0 to N-1)
$for (i, 100) {
    // i goes from 0 to 99
    process(i);
};

// Range-based for loop (start to end-1)
$for (i, 10, 20) {
    // i goes from 10 to 19
};

// Range-based for loop with step
$for (i, 0, 100, 2) {
    // i goes 0, 2, 4, ..., 98
};

// Infinite loop
$loop {
    // Runs forever unless broken
    $if (should_exit) {
        $break;
    };
};
```

### Switch Statements

```cpp
$switch (value) {
    $case (1) {
        result = "one";
    };
    $case (2) {
        result = "two";
    };
    $default {
        result = "other";
    };
};
```

> **Note**: Unlike C++, `$break` is automatically inserted at the end of each case. Fall-through is not supported.

### Control Flow Modifiers

```cpp
// Break out of loop
$while (true) {
    $if (condition) {
        $break;
    };
};

// Continue to next iteration
$for (i, 100) {
    $if (i % 2 == 0) {
        $continue;  // Skip even numbers
    };
    process(i);
};

// Early return from callable
Callable early_exit = [](Float x) noexcept {
    $if (x < 0.0f) {
        $return(-x);  // Return from callable
    };
    return x;
};
```

## Multi-Stage Programming

A key feature of LuisaCompute is the ability to mix host-side (C++) and device-side (DSL) control flow:

```cpp
// Host-side constant
constexpr bool use_optimization = true;

Kernel2D my_kernel = [&](ImageFloat img) noexcept {
    Var coord = dispatch_id().xy();
    
    // Host-side if: evaluated during kernel tracing
    // Only one branch is recorded in the AST
    if constexpr (use_optimization) {
        // This code is only in the AST if use_optimization is true
        auto fast_result = optimized_path();
        img->write(coord, fast_result);
    } else {
        auto slow_result = reference_path();
        img->write(coord, slow_result);
    }
    
    // Device-side if: both branches recorded, GPU decides at runtime
    Var<float> threshold = 0.5f;
    $if (threshold > 0.5f) {
        // Both branches are in the AST
        img->write(coord, red);
    } $else {
        img->write(coord, blue);
    };
};
```

This enables:
- **Template-based kernel specialization**
- **Compile-time configuration**
- **Conditional feature compilation**

## Callables

Callables are reusable device functions that can be called from kernels or other callables.

### Basic Callable

```cpp
// Define a simple callable
Callable add = [](Float a, Float b) noexcept {
    return a + b;
};

// Use in a kernel
Kernel1D my_kernel = [&](BufferFloat buffer) noexcept {
    Var i = dispatch_id().x;
    Var x = buffer.read(i);
    buffer.write(i, add(x, 1.0f));
};
```

### Callable with Multiple Return Values

```cpp
Callable split_rgb = [](Float4 rgba) noexcept {
    return make_tuple(rgba.r(), rgba.g(), rgba.b());
};

Kernel2D process = [&](ImageFloat image) noexcept {
    Var coord = dispatch_id().xy();
    Float4 pixel = image->read(coord);
    
    // Unpack multiple returns
    auto [r, g, b] = split_rgb(pixel);
    Float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    
    image->write(coord, make_float4(gray, gray, gray, 1.0f));
};
```

### Callable Limitations

**Important:** Callables have the following restrictions:

1. **No Recursion**: Callables cannot call themselves or be part of recursive call chains. The AST is constructed statically during kernel definition, and recursion requires dynamic call stacks that are not supported by GPU execution models.

2. **No Dynamic Dispatch**: The callable to be invoked must be known at compile time. You cannot select callables at runtime based on device-side conditions.

3. **Resource Capture**: Callables can capture buffers, images, and other resources by reference, but these captures happen at kernel definition time (host-side), not at kernel execution time.

```cpp
// GOOD: Non-recursive callable
Callable factorial_iter = [](Int n) noexcept {
    Var<int> result = 1;
    $for (i, 1, n + 1) {
        result = result * i;
    };
    return result;
};

// BAD: Recursive callable - NOT SUPPORTED
// Callable factorial_recursive = [](Int n) noexcept {
//     $if (n <= 1) {
//         $return(1);
//     };
//     return n * factorial_recursive(n - 1);  // ERROR!
// };
```

## Kernels

Kernels are entry points for GPU execution. They define the parallel grid dimension.

### Kernel Types

```cpp
// 1D kernel - for linear processing
Kernel1D process_array = [&](BufferFloat data) noexcept {
    Var idx = dispatch_id().x;
    // Process element at idx
};

// 2D kernel - for image processing
Kernel2D process_image = [&](ImageFloat image) noexcept {
    Var coord = dispatch_id().xy();
    // Process pixel at coord
};

// 3D kernel - for volume processing
Kernel3D process_volume = [&](VolumeFloat volume) noexcept {
    Var coord = dispatch_id().xyz();
    // Process voxel at coord
};
```

### Dispatch Coordinates

```cpp
Kernel2D render = [&](ImageFloat image) noexcept {
    // Thread position in the dispatch grid
    UInt2 coord = dispatch_id().xy();
    
    // Total size of the dispatch grid
    UInt2 size = dispatch_size().xy();
    
    // Thread position within its block
    UInt2 local = thread_id().xy();
    
    // Block position in the grid
    UInt2 block = block_id().xy();
    
    // Block size (threads per block)
    UInt2 block_size = block_size().xy();
    
    // Normalized UV coordinates
    Float2 uv = (make_float2(coord) + 0.5f) / make_float2(size);
};
```

### Custom Block Size

The block size (threads per block) can be set inside the kernel using `set_block_size()`:

```cpp
Kernel2D render = [&](ImageFloat image) noexcept {
    // Set block size to 32x32 = 1024 threads per block
    set_block_size(32u, 32u);
    
    Var coord = dispatch_id().xy();
    // ... kernel body ...
};

auto shader = device.compile(render);
```

If not specified, default block sizes are used:
- 1D kernels: 256 threads per block
- 2D kernels: 16x16 threads per block  
- 3D kernels: 8x8x8 threads per block

> **Note:** The total threads per block should not exceed the hardware limit (typically 1024 for CUDA/Metal/DirectX).

## Built-in Functions

### Mathematical Functions

```cpp
// Basic arithmetic
Float c = max(a, b);
Float c = min(a, b);
Float c = clamp(a, 0.0f, 1.0f);  // Clamp to [0, 1]
Float c = saturate(a);            // Same as clamp(a, 0, 1)
Float c = abs(a);
Float c = sign(a);                // -1, 0, or 1

// Rounding
Float c = floor(a);
Float c = ceil(a);
Float c = round(a);
Float c = trunc(a);
Float c = fract(a);               // Fractional part

// Exponential and logarithmic
Float c = pow(a, b);              // a^b
Float c = exp(a);
Float c = log(a);
Float c = log2(a);
Float c = exp2(a);
Float c = sqrt(a);
Float c = rsqrt(a);               // 1/sqrt(a), often faster

// Trigonometric (angles in radians)
Float c = sin(a);
Float c = cos(a);
Float c = tan(a);
Float c = asin(a);
Float c = acos(a);
Float c = atan(a);
Float c = atan2(y, x);            // Four-quadrant arctangent
Float c = sinh(a);                // Hyperbolic sine
Float c = cosh(a);                // Hyperbolic cosine

// Interpolation
Float c = lerp(a, b, t);          // Linear: a + t*(b-a)
Float c = smoothstep(edge0, edge1, x);  // Smooth Hermite interpolation
Float c = step(edge, x);          // 0 if x < edge, else 1

// Vector operations
Float len = length(v);
Float3 n = normalize(v);
Float dot_prod = dot(a, b);
Float3 cross_prod = cross(a, b);
Float dist = distance(a, b);
Float3 refl = reflect(incident, normal);
Float3 refr = refract(incident, normal, eta);

// Component-wise operations
Float3 c = fma(a, b, c);          // a*b + c (fused multiply-add)
Float3 c = select(a, b, cond);    // cond ? b : a
```

### Type Conversions

```cpp
// Static cast (value conversion)
Float f = 3.7f;
Int i = cast<int>(f);             // 3 (truncation)

// Bitwise reinterpretation
Float f = 1.0f;
UInt bits = as<uint>(f);          // 0x3f800000

// Matrix transpose
Float3x3 m = ...;
Float3x3 mt = transpose(m);

// Matrix inverse
Float3x3 inv = inverse(m);

// Matrix determinant
Float det = determinant(m);
```

## Automatic Differentiation

LuisaCompute supports reverse-mode automatic differentiation:

```cpp
Kernel1D gradient_descent = [&](BufferFloat x, BufferFloat y, BufferFloat grad_x) noexcept {
    Var idx = dispatch_id().x;
    
    $autodiff {
        // Mark variables that need gradients
        requires_grad(x);
        
        // Forward pass
        Var<float> xi = x.read(idx);
        Var<float> yi = y.read(idx);
        Var<float> loss = (xi - yi) * (xi - yi);  // (x - y)^2
        
        // Backward pass
        backward(loss);
        
        // Get gradients
        grad_x.write(idx, grad(xi));
    };
};
```

### Autodiff with Control Flow

```cpp
Callable complex_func = [](Float x, Float y) noexcept {
    Var<float> result;
    $if (x > 0.0f) {
        result = x * y;
    } $else {
        result = x + y;
    };
    return result;
};

Kernel1D compute_grad = [&](BufferFloat x_buf, BufferFloat y_buf, 
                            BufferFloat dx, BufferFloat dy) noexcept {
    Var idx = dispatch_id().x;
    
    $autodiff {
        Var<float> x = x_buf.read(idx);
        Var<float> y = y_buf.read(idx);
        requires_grad(x, y);
        
        Var<float> z = complex_func(x, y);
        backward(z);
        
        dx.write(idx, grad(x));
        dy.write(idx, grad(y));
    };
};
```

**Limitations:**
- Loops with dynamic iteration counts must be unrolled manually
- Some operations may not be differentiable

## Best Practices

### 1. Prefer Expr<T> for Temporary Values

```cpp
// Creates unnecessary variable
Float temp = a + b;
Float result = temp * c;

// More efficient
Float result = (Expr{a} + b) * c;
```

### 2. Use Callables for Code Reuse

```cpp
// Bad: duplicated code in each kernel
Kernel2D kernel1 = [&](ImageFloat img) noexcept {
    // complex calculation...
};
Kernel2D kernel2 = [&](ImageFloat img) noexcept {
    // same complex calculation...
};

// Good: shared callable
Callable complex_calc = [](Float2 uv) noexcept {
    // implementation
    return result;
};
```

### 3. Minimize Divergence

```cpp
// Bad: divergent branches within warps
$if (thread_id().x % 2 == 0) {
    do_a();
} $else {
    do_b();
};

// Better: try to align branching with warp boundaries
```

### 4. Use Native C++ for Compile-Time Decisions

```cpp
// Good: compile-time specialization
if constexpr (use_fast_path) {
    // Fast code path
} else {
    // Accurate code path
}

// Only use $if for runtime decisions
$if (runtime_condition) {
    // Runtime branching
};
```

### 5. Resource Capture vs Parameters

```cpp
// Capture for resources that don't change per dispatch
BufferFloat constant_buffer = ...;
ImageFloat output_image = ...;

Kernel2D render = [&](Float time) noexcept {
    // time varies per frame
    // constant_buffer and output_image captured by reference
};

// Pass as parameter for resources that vary per dispatch
Kernel2D process = [&](BufferFloat input, BufferFloat output) noexcept {
    // Different buffers each dispatch
};
```
