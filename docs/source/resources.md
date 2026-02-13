# Resources and Runtime

This document covers the LuisaCompute runtime system, including resource management, command execution, and synchronization.

## Overview

The LuisaCompute runtime provides a unified abstraction layer over various GPU APIs (CUDA, DirectX, Metal). It handles:

- **Resource Management**: Buffers, images, volumes, and acceleration structures
- **Command Scheduling**: Asynchronous execution with automatic dependency tracking
- **Memory Transfers**: Efficient data movement between host and device
- **Synchronization**: Stream and event-based coordination

## Context

The `Context` is the entry point to the LuisaCompute runtime. It manages backend plugins and global configuration.

### Creating a Context

```cpp
#include <luisa/luisa-compute.h>

// From command-line arguments
int main(int argc, char* argv[]) {
    Context context{argv[0]};
    // ...
}

// From explicit path
Context context{"/path/to/executable"};

// With custom data directory (for caches)
Context context{"/path/to/executable", "/path/to/data"};
```

### Querying Available Backends

```cpp
Context context{argv[0]};

// List installed backends
auto backends = context.installed_backends();
for (auto& name : backends) {
    std::cout << "Backend: " << name << std::endl;
}

// Query device names for a backend
auto cuda_devices = context.backend_device_names("cuda");
for (auto& name : cuda_devices) {
    std::cout << "CUDA device: " << name << std::endl;
}
```

## Device

A `Device` represents a physical or virtual GPU/CPU for computation.

### Creating Devices

```cpp
// Create specific backend
Device cuda = context.create_device("cuda");
Device dx = context.create_device("dx");           // Windows
Device metal = context.create_device("metal");     // macOS
Device cpu = context.create_device("cpu");         // Any platform

// Create with validation enabled
Device device = context.create_device("cuda", nullptr, true);

// Or via environment variable
// LUISA_ENABLE_VALIDATION=1 ./program

// Use first available backend
Device device = context.create_default_device();
```

### Device Properties

```cpp
// Query device information
std::string backend = device.backend_name();       // "cuda", "dx", etc.
void* native_handle = device.native_handle();      // CUcontext, ID3D12Device*, etc.
uint warp_size = device.compute_warp_size();       // SIMD width (e.g., 32 for CUDA)
size_t granularity = device.memory_granularity();  // Allocation alignment

// Check if device is valid
if (device) {
    // Device is properly initialized
}
```

### Backend Extensions

Devices may support backend-specific extensions:

```cpp
// CUDA-specific extensions
if (auto cuda_ext = device.extension<CUDAExternalExt>()) {
    // Use CUDA interop features
}

// DirectX-specific extensions
if (auto dx_ext = device.extension<DXConfigExt>()) {
    // Use DirectX-specific features
}
```

## Buffers

Buffers are linear memory regions for structured data storage.

### Creating Buffers

```cpp
// Simple buffer of 1000 floats
Buffer<float> float_buf = device.create_buffer<float>(1000);

// Buffer of vectors
Buffer<float4> vec4_buf = device.create_buffer<float4>(1024);

// Buffer of custom structures (must be reflected with LUISA_STRUCT)
struct Particle {
    float3 position;
    float3 velocity;
    float mass;
};
LUISA_STRUCT(Particle, position, velocity, mass) {};

Buffer<Particle> particles = device.create_buffer<Particle>(10000);
```

### Buffer Operations

```cpp
// Upload from host
std::vector<float> host_data(1000, 1.0f);
stream << float_buf.copy_from(host_data.data());

// Download to host
std::vector<float> result(1000);
stream << float_buf.copy_to(result.data());

// Copy between buffers
Buffer<float> other_buf = device.create_buffer<float>(1000);
stream << other_buf.copy_from(float_buf);

// Fill with value (in a kernel)
Kernel1D fill = [&](BufferFloat buf, Float value) noexcept {
    Var i = dispatch_id().x;
    buf.write(i, value);
};
auto fill_shader = device.compile(fill);
stream << fill_shader(float_buf, 3.14f).dispatch(1000);
```

### Sub-Buffers

```cpp
// Create a view into a portion of a buffer
Buffer<float> large_buf = device.create_buffer<float>(10000);

// View elements [1000, 5999]
BufferView<float> sub = large_buf.view(1000, 5000);

// Use sub-buffer in kernels
stream << kernel(sub).dispatch(5000);

// Copy to/from sub-buffers
stream << sub.copy_from(host_data.data());
```

### Byte Buffers

For untyped/raw memory access:

```cpp
// Create byte buffer (1000 bytes)
ByteBuffer raw = device.create_byte_buffer(1000);

// Use in kernels with manual pointer arithmetic
Kernel1D process = [&](ByteBuffer raw) noexcept {
    Var idx = dispatch_id().x;
    // Access via buffer pointer...
};
```

## Images

Images are 2D textures with hardware-accelerated sampling and caching.

### Creating Images

```cpp
// Basic RGBA8 image (1024x1024)
// Template type determines how pixels are read/written
// Storage format determines internal memory layout
Image<float> image = device.create_image<float>(
    PixelStorage::BYTE4,    // 4 channels, 8 bits each
    1024, 1024              // Width, height
);

// Floating point format
Image<float> hdr_image = device.create_image<float>(
    PixelStorage::FLOAT4,   // 4 channels, 32-bit float each
    1920, 1080
);

// With mipmaps
Image<float> mip_image = device.create_image<float>(
    PixelStorage::BYTE4,
    1024, 1024,
    10                      // 10 mipmap levels (1024 -> 1)
);

// Simultaneous access (for concurrent read/write)
Image<float> shared = device.create_image<float>(
    PixelStorage::BYTE4,
    1024, 1024, 1,          // 1 mipmap level
    true                    // Simultaneous access
);
```

### Pixel Storage Formats

| Storage Format | Channels | Bits per Channel | C++ Read/Write Type |
|---------------|----------|------------------|---------------------|
| `BYTE1` | 1 | 8 | `float` |
| `BYTE2` | 2 | 8 | `float2` |
| `BYTE4` | 4 | 8 | `float4` |
| `SHORT1` | 1 | 16 | `float` |
| `SHORT2` | 2 | 16 | `float2` |
| `SHORT4` | 4 | 16 | `float4` |
| `FLOAT1` | 1 | 32 | `float` |
| `FLOAT2` | 2 | 32 | `float2` |
| `FLOAT4` | 4 | 32 | `float4` |

### Image Operations

```cpp
// Upload from host
std::vector<std::byte> pixels(width * height * 4);
// ... fill pixels ...
stream << image.copy_from(pixels.data());

// Download to host
stream << image.copy_to(pixels.data());

// Copy between images
Image<float> dest = device.create_image<float>(PixelStorage::BYTE4, 1024, 1024);
stream << dest.copy_from(image);

// Generate mipmaps
stream << image.generate_mipmaps();
```

### Image Views

```cpp
// View a specific mipmap level
ImageView<float> level0 = image.view(0);  // Full resolution
ImageView<float> level1 = image.view(1);  // Half resolution

// View in kernel
Kernel2D process = [&](ImageFloat image) noexcept {
    Var coord = dispatch_id().xy();
    Float4 color = image.read(coord);
    image.write(coord, color * 0.5f);
};
stream << process(image.view(0)).dispatch(1024, 1024);
```

## Volumes

Volumes are 3D textures, useful for volumetric data and 3D lookups.

```cpp
// Create 256^3 volume
Volume<float> volume = device.create_volume<float>(
    PixelStorage::BYTE4,
    256, 256, 256
);

// Upload 3D data
std::vector<std::byte> voxel_data(256 * 256 * 256 * 4);
stream << volume.copy_from(voxel_data.data());

// Use in kernel
Kernel3D process = [&](VolumeFloat vol) noexcept {
    Var coord = dispatch_id().xyz();
    Float4 value = vol.read(coord);
    vol.write(coord, value * 2.0f);
};
stream << process(volume).dispatch(256, 256, 256);
```

## Bindless Arrays

Bindless arrays allow dynamic indexing of resources in shaders, bypassing binding limits.

### Creating Bindless Arrays

```cpp
// Create with 65536 slots (default)
BindlessArray bindless = device.create_bindless_array();

// Or specify slot count
BindlessArray bindless = device.create_bindless_array(1024);

// Specify allowed resource types
BindlessArray textures_only = device.create_bindless_array(
    1024,
    BindlessSlotType::TEXTURE_ONLY  // Only images/volumes
);
```

### Using Bindless Arrays

```cpp
// Prepare resources
std::vector<Image<float>> textures;
for (int i = 0; i < 100; i++) {
    textures.push_back(device.create_image<float>(PixelStorage::BYTE4, 512, 512));
}

// Bind resources to slots
luisa::vector<BindlessArrayUpdate> updates;
for (int i = 0; i < textures.size(); i++) {
    updates.emplace_back(BindlessArrayUpdate{
        BindlessArrayUpdate::Operation::Emplace,
        i,                      // Slot index
        textures[i].handle(),   // Resource handle
        Sampler::linear_linear_mirror()  // Sampler
    });
}

// Apply updates
stream << bindless.update(updates);

// Use in kernel
Kernel2D sample_textures = [&](BindlessArray textures, Int tex_id) noexcept {
    Var coord = dispatch_id().xy();
    Float2 uv = make_float2(coord) / 512.0f;
    
    // Sample from dynamically selected texture
    Float4 color = textures.sample(tex_id, uv);
    // ...
};
```

## Acceleration Structures (Ray Tracing)

For hardware-accelerated ray-scene intersection.

### Meshes

```cpp
// Create mesh from vertex and index buffers
Buffer<float3> vertices = device.create_buffer<float3>(num_vertices);
Buffer<Triangle> triangles = device.create_buffer<Triangle>(num_triangles);
// ... fill buffers ...

Mesh mesh = device.create_mesh(vertices, triangles);

// Build acceleration structure
stream << mesh.build();
```

### Acceleration Structure (Accel)

```cpp
// Create top-level acceleration structure
Accel accel = device.create_accel();

// Add meshes to the accel
luisa::vector<Accel::Modification> mods;
mods.push_back(Accel::Modification{
    Accel::Modification::operation_set_mesh,
    mesh.handle(),      // Bottom-level mesh
    float4x4::identity(), // Transform matrix
    0,                  // Instance ID (visible in shader)
    0xff,               // Visibility mask
    0,                  // Instance index
    Accel::Modification::flag_mesh
});

// Build/update accel
stream << accel.build(mods);

// Use in ray tracing kernel
Kernel2D trace = [&](ImageFloat image, Accel accel) noexcept {
    Var coord = dispatch_id().xy();
    
    Ray ray = ...;
    Var<hit> hit = accel.intersect(ray, {});
    
    $if (hit->hit()) {
        // Process hit
    };
};
```

## Streams

Streams are command queues for asynchronous execution.

### Creating Streams

```cpp
// Compute stream (default)
Stream compute = device.create_stream(StreamTag::COMPUTE);

// Graphics stream (for rasterization)
Stream graphics = device.create_stream(StreamTag::GRAPHICS);

// Copy/streaming stream
Stream transfer = device.create_stream(StreamTag::COPY);
```

### Submitting Commands

```cpp
// Simple command sequence
stream << buffer.copy_from(host_data)
       << shader.dispatch(1024, 1024)
       << buffer.copy_to(result_data)
       << synchronize();

// Complex command buffer
{
    auto cmd_list = stream.command_buffer();
    cmd_list << buffer_a.copy_from(data_a)
             << buffer_b.copy_from(data_b)
             << compute_shader(buffer_a, buffer_b)
                 .dispatch(1024, 1024)
             << buffer_c.copy_from(buffer_a)
             << commit();
}

// Host callbacks
stream << shader.dispatch(1024, 1024)
       << [&]() {
           // This runs on host after shader completes
           std::cout << "Shader finished!" << std::endl;
       }
       << synchronize();
```

### Stream Synchronization

```cpp
// Synchronize specific stream
stream.synchronize();
stream << synchronize();

// Wait for specific event
Event event = device.create_event();
stream << shader.dispatch(1024, 1024)
       << event.signal()
       << synchronize();

event.synchronize();  // Block until signaled
```

## Events

Events provide synchronization between streams or with the host.

### Basic Usage

```cpp
Event event = device.create_event();

// Stream A signals
Stream stream_a = device.create_stream();
stream_a << command_a
         << event.signal()
         << synchronize();

// Stream B waits
Stream stream_b = device.create_stream();
stream_b << event.wait()
         << command_b  // Executes after command_a completes
         << synchronize();
```

### Timeline Events

For more complex synchronization patterns:

```cpp
TimelineEvent timeline = device.create_timeline_event();

// Signal with value
stream_a << timeline.signal(100);

// Wait for specific value
stream_b << timeline.wait(100);  // Wait until value >= 100
```

## Memory Model and RAII

All resources follow RAII (Resource Acquisition Is Initialization):

```cpp
{
    // Resource created
    Buffer<float> buffer = device.create_buffer<float>(1000);
    Image<float> image = device.create_image<float>(PixelStorage::BYTE4, 1024, 1024);
    
    // Use resources...
    
} // Resources automatically released here
```

### Move Semantics

Resources are move-only (cannot be copied):

```cpp
Buffer<float> buf1 = device.create_buffer<float>(1000);
Buffer<float> buf2 = std::move(buf1);  // Valid
// Buffer<float> buf3 = buf2;          // Error: copy not allowed
```

### Resource Lifetimes

Important: Ensure resources outlive their commands:

```cpp
void problematic(Stream& stream) {
    Buffer<float> local = device.create_buffer<float>(1000);
    stream << kernel(local).dispatch(1000);
    // local destroyed here, but command may not have executed!
}

// Better
void correct(Stream& stream, Buffer<float>& buffer) {
    stream << kernel(buffer).dispatch(1000);
}  // Caller keeps buffer alive
```

## Command Scheduling

LuisaCompute automatically analyzes command dependencies and optimizes execution order:

```cpp
// These commands have no dependencies - may execute in parallel
stream << buf_a.copy_from(host_a)    // Write to buf_a
       << buf_b.copy_from(host_b)    // Write to buf_b (independent)
       << kernel(buf_a).dispatch(n)   // Read buf_a
       << kernel(buf_b).dispatch(n)   // Read buf_b (can run concurrently)
       << synchronize();

// Dependencies are automatically tracked
stream << buf.write(data)            // Write to buf
       << kernel(buf).dispatch(n)    // Read buf - waits for write
       << buf.copy_to(result)        // Read buf - waits for kernel
       << synchronize();
```

## Best Practices

### 1. Batch Commands

```cpp
// Good: Fewer, larger command batches
stream << cmd1 << cmd2 << cmd3 << cmd4 << synchronize();

// Avoid: Many small submissions
stream << cmd1 << synchronize();
stream << cmd2 << synchronize();
```

### 2. Reuse Resources

```cpp
// Good: Allocate once, reuse
Buffer<float> buffer = device.create_buffer<float>(max_size);
for (int frame = 0; frame < 1000; frame++) {
    stream << update_buffer(buffer)
           << process(buffer)
           << synchronize();
}

// Avoid: Reallocating each frame
for (int frame = 0; frame < 1000; frame++) {
    Buffer<float> buffer = device.create_buffer<float>(size);
    // ...
}
```

### 3. Use Appropriate Stream Types

```cpp
// Compute work on compute stream
Stream compute = device.create_stream(StreamTag::COMPUTE);

// Transfer work on copy stream (may overlap with compute)
Stream transfer = device.create_stream(StreamTag::COPY);

transfer << upload_data(buffer);
compute << process(buffer);
// Upload and compute can overlap!
```

### 4. Minimize Host-GPU Transfers

```cpp
// Good: Keep data on GPU
Buffer<float> persistent = device.create_buffer<float>(n);
for (int i = 0; i < 100; i++) {
    stream << kernel(persistent).dispatch(n);
}
stream << persistent.copy_to(host);

// Avoid: Round-trips
for (int i = 0; i < 100; i++) {
    stream << buffer.copy_from(host_data)  // Upload
           << kernel(buffer).dispatch(n)
           << buffer.copy_to(host_data)    // Download
           << synchronize();
}
```
