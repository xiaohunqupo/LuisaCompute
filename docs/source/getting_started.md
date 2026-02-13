# Getting Started

This guide will help you get up and running with LuisaCompute, from installation to your first GPU-accelerated program.

## Overview

LuisaCompute is a high-performance cross-platform computing framework designed for modern graphics and general-purpose GPU computing. It balances three key goals:

- **Unification**: Write code once, run on CUDA, DirectX, Metal, or CPU
- **Programmability**: Modern C++ embedded DSL with intuitive syntax
- **Performance**: Optimized backends with automatic command scheduling

The framework consists of three major components:
1. **Embedded DSL**: Write GPU kernels directly in C++ using `Var<T>`, `Kernel`, and `Callable`
2. **Unified Runtime**: Resource management and command scheduling with `Context`, `Device`, and `Stream`
3. **Multiple Backends**: CUDA, DirectX, Metal, and CPU implementations

## Prerequisites

### Hardware Requirements

| Backend | Requirements |
|---------|-------------|
| CUDA | NVIDIA GPU with RTX support (RTX 20 series or newer recommended) |
| DirectX | DirectX 12.1 & Shader Model 6.5 compatible GPU |
| Metal | macOS device with Metal support |
| CPU | Any modern x86_64 or ARM64 processor |

### Software Requirements

- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS 12+
- **Compiler**: C++20 compatible compiler (MSVC 2019+, GCC 11+, Clang 14+)
- **Build System**: CMake 3.20+ or XMake 2.7+
- **Python**: 3.10+ (for bootstrap script and Python frontend)

#### Backend-Specific Requirements

- **CUDA**: CUDA Toolkit 11.7+, NVIDIA driver R535+
- **DirectX**: Windows SDK 10.0.19041.0+
- **Metal**: Xcode 14+ (macOS)

## Building from Source

### Clone the Repository

```bash
git clone -b next https://github.com/LuisaGroup/LuisaCompute.git --recursive
cd LuisaCompute
```

> **Important**: The `--recursive` flag is required to fetch all submodules.

### Build Using the Bootstrap Script (Recommended)

The bootstrap script automates dependency installation and building:

```bash
# Build with CUDA backend using CMake
python bootstrap.py cmake -f cuda -b

# Build with all available backends
python bootstrap.py cmake -f all -b

# Build with specific configuration
python bootstrap.py cmake -f cuda -b -- -DCMAKE_BUILD_TYPE=Release

# Generate IDE project without building
python bootstrap.py cmake -f cuda -c -o cmake-build-release
```

Install missing dependencies:
```bash
python bootstrap.py -i rust      # Install Rust (for CPU backend)
python bootstrap.py -i cmake     # Install/upgrade CMake
python bootstrap.py -i xmake     # Install XMake
```

### Build Using CMake Directly

```bash
mkdir build && cd build
cmake .. -DLUISA_BACKEND_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

### Build Using XMake

```bash
xmake config --cuda=true
xmake build
```

### CMake Integration in Your Project

```cmake
# Add LuisaCompute as a subdirectory
add_subdirectory(LuisaCompute)

# Link to your target
target_link_libraries(your_target PRIVATE luisa::compute)
```

Or use the [starter template](https://github.com/LuisaGroup/CMakeStarterTemplate) for quick setup.

## Your First Program

Let's create a simple program that fills an image with a gradient color on the GPU.

### Complete Example

```cpp
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>  // For $if, $while, etc.
#include <stb_image_write.h>   // For saving the result

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    // Step 1: Initialize the runtime context
    Context context{argv[0]};
    
    // Step 2: Create a device (GPU) to run computations on
    Device device = context.create_device("cuda");
    
    // Step 3: Create a command stream for submitting work
    Stream stream = device.create_stream();
    
    // Step 4: Create a 1024x1024 image with 4-channel 8-bit storage
    // The template argument <float> means pixel values are automatically
    // converted between byte4 and float4 when reading/writing
    Image<float> image = device.create_image<float>(
        PixelStorage::BYTE4,  // Internal storage format
        1024, 1024            // Width and height
    );
    
    // Step 5: Define a callable (reusable function)
    Callable linear_to_srgb = [](Float4 linear) noexcept {
        Float3 rgb = linear.xyz();
        // Apply gamma correction
        $if (rgb <= 0.00031308f) {
            rgb = 12.92f * rgb;
        } $else {
            rgb = 1.055f * pow(rgb, 1.0f / 2.4f) - 0.055f;
        };
        return make_float4(rgb, linear.w);
    };
    
    // Step 6: Define a 2D kernel (entry point for GPU execution)
    Kernel2D fill_kernel = [&](ImageFloat image) noexcept {
        // Get the current thread's position in the grid
        Var coord = dispatch_id().xy();
        Var size = dispatch_size().xy();
        
        // Calculate normalized UV coordinates (0 to 1)
        Float2 uv = make_float2(coord) / make_float2(size);
        
        // Create a gradient color
        Float4 color = make_float4(uv, 0.5f, 1.0f);
        
        // Apply sRGB conversion and write to image
        image->write(coord, linear_to_srgb(color));
    };
    
    // Step 7: Compile the kernel into a shader
    auto shader = device.compile(fill_kernel);
    
    // Step 8: Prepare host memory for downloading the result
    std::vector<std::byte> pixels(1024 * 1024 * 4);
    
    // Step 9: Execute commands on the GPU
    stream << shader(image.view(0)).dispatch(1024, 1024)  // Launch kernel
           << image.copy_to(pixels.data())                 // Download to CPU
           << synchronize();                               // Wait for completion
    
    // Step 10: Save the result
    stbi_write_png("output.png", 1024, 1024, 4, pixels.data(), 0);
    
    return 0;
}
```

### Building Your Program

```bash
# Compile with your build system, linking against LuisaCompute
g++ -std=c++20 myprogram.cpp -I/path/to/luisa/include \
    -L/path/to/luisa/lib -lluisa-compute-runtime \
    -o myprogram
```

## Core Concepts

### 1. Context and Device

The **Context** is the entry point for the runtime. It manages backend plugins and global configuration:

```cpp
// Create context with executable path
Context context{argv[0]};

// Create a specific backend device
Device cuda_device = context.create_device("cuda");
Device dx_device = context.create_device("dx");     // Windows only
Device metal_device = context.create_device("metal"); // macOS only
Device cpu_device = context.create_device("cpu");

// Or use the default available backend
Device device = context.create_default_device();
```

### 2. Streams and Commands

**Streams** are command queues for asynchronous execution:

```cpp
// Create a compute stream
Stream stream = device.create_stream(StreamTag::COMPUTE);

// Commands are submitted using operator<<
stream << command1 << command2 << command3;

// Commands are automatically batched and executed
// Explicit synchronization:
stream << synchronize();
// Or:
stream.synchronize();
```

### 3. Resources

#### Buffers

Linear memory for structured data:

```cpp
// Create a buffer of 1000 float4 elements
Buffer<float4> buffer = device.create_buffer<float4>(1000);

// Upload data from host
std::vector<float4> host_data(1000);
stream << buffer.copy_from(host_data.data());

// Download data to host
stream << buffer.copy_to(host_data.data());
```

#### Images

2D textures with hardware-accelerated sampling:

```cpp
// Create a 1920x1080 image with RGBA8 format
Image<float> image = device.create_image<float>(
    PixelStorage::BYTE4, 1920, 1080);

// Create with mipmaps
Image<float> image_with_mips = device.create_image<float>(
    PixelStorage::BYTE4, 1920, 1080, 10);  // 10 mipmap levels
```

### 4. The DSL

#### Variables

Use `Var<T>` to represent device-side variables:

```cpp
// Scalar types
Var<float> x = 1.0f;           // Or: Float x = 1.0f;
Var<int> count = 100;          // Or: Int count = 100;

// Vector types
Float2 pos = make_float2(1.0f, 2.0f);
Float3 color = make_float3(1.0f, 0.5f, 0.0f);
Float4 rgba = make_float4(color, 1.0f);

// Swizzling
Float2 xy = pos.xy();          // Extract x and y
Float3 xyz = rgba.xyz();       // Extract first three components
Float4 repeated = pos.xxxx();  // (x, x, x, x)
```

#### Control Flow

Use `$`-prefixed macros for device-side control flow:

```cpp
$if (condition) {
    // GPU-side branch
} $elif (other_condition) {
    // Another branch
} $else {
    // Default branch
};

$while (condition) {
    // Loop on GPU
};

$for (i, 0, 100) {
    // Loop from 0 to 99
    // i is Var<int>
};

$for (i, 0, 100, 2) {
    // Loop from 0 to 98 with step 2
};
```

> **Note**: Use native C++ `if` for host-side (compile-time) decisions that affect what gets recorded into the kernel.

#### Kernels and Callables

```cpp
// A Callable is a reusable device function
Callable lerp = [](Float a, Float b, Float t) noexcept {
    return a * (1.0f - t) + b * t;
};

// A Kernel is the entry point for GPU execution
Kernel2D render_kernel = [&](ImageFloat image, BufferFloat4 colors) noexcept {
    Var coord = dispatch_id().xy();
    Var index = coord.y * dispatch_size().x + coord.x;
    
    Float4 color = colors.read(index);
    image->write(coord, color);
};

// Compile and dispatch
auto shader = device.compile(render_kernel);
stream << shader(image, colors).dispatch(1024, 1024);
```

## Next Steps

- Learn more about the [DSL](dsl.md) for advanced kernel programming
- Explore [Resources and Runtime](resources.md) for detailed resource management
- Understand the [Architecture](architecture.md) for implementation details
- Check out the [API Reference](api_reference.rst) for complete class documentation
- Browse example programs in `src/tests` for real-world usage patterns

## Troubleshooting

### Backend Not Found

```
Error: Backend 'cuda' not found
```

- Ensure the backend plugin exists: `luisa-backend-cuda.dll` (Windows) or `luisa-backend-cuda.so` (Linux)
- Check that the plugin is in the same directory as your executable
- Verify CUDA toolkit is installed and drivers are up to date

### Compilation Errors

- Ensure C++20 is enabled (`-std=c++20` for GCC/Clang, `/std:c++20` for MSVC)
- Check that all submodules are cloned (`git submodule update --init --recursive`)

### Runtime Validation

Enable validation layer for debugging:
```cpp
Device device = context.create_device("cuda", nullptr, true);
// Or via environment variable:
// LUISA_ENABLE_VALIDATION=1 ./myprogram
```
