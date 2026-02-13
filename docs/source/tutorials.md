# Tutorials

This section provides step-by-step tutorials for building practical applications with LuisaCompute.

## Table of Contents

1. [ShaderToy-Style Mandelbrot](#shadertoy-style-mandelbrot-set) - A gentle introduction
2. [Path Tracing Renderer](#path-tracing-renderer) - Global illumination with ray tracing
3. [MPM Fluid Simulation](#mpm-fluid-simulation) - Material Point Method for fluids
4. [Conway's Game of Life](#conways-game-of-life) - Cellular automata with ping-pong buffering
5. [Wave Equation Simulation](#wave-equation-simulation) - Interactive PDE solver
6. [Image Processing Pipeline](#image-processing-pipeline) - Multi-pass filters and effects
7. [N-Body Simulation](#n-body-gravitational-simulation) - Gravitational particle physics

---

## ShaderToy-Style Mandelbrot Set

Let's start with a simple yet visually appealing example: rendering the Mandelbrot set. This mimics the style of [ShaderToy](https://www.shadertoy.com/), a popular platform for fragment shaders.

### Final Result

A colorful visualization of the Mandelbrot set with smooth coloring.

### Step-by-Step Implementation

#### Step 1: Project Setup

Create a new project with the following structure:

```
mandelbrot/
├── CMakeLists.txt
└── main.cpp
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.23)
project(mandelbrot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Add LuisaCompute
add_subdirectory(LuisaCompute)

add_executable(mandelbrot main.cpp)
target_link_libraries(mandelbrot PRIVATE luisa::compute)
```

#### Step 2: Basic Structure

**main.cpp - Initial Setup:**

```cpp
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    // Step 1: Create context and device
    Context context{argv[0]};
    Device device = context.create_device("cuda");  // or "cpu", "metal", "dx"
    Stream stream = device.create_stream();
    
    // Step 2: Create an image to render to
    static constexpr uint2 resolution = make_uint2(1024u, 1024u);
    Image<float> image = device.create_image<float>(
        PixelStorage::FLOAT4, resolution.x, resolution.y);
    
    // We'll add the kernel here...
    
    return 0;
}
```

#### Step 3: Understanding the Mandelbrot Set

The Mandelbrot set is defined by the iteration:
```
z_{n+1} = z_n² + c
```

Where `c` is a complex number corresponding to pixel coordinates. If the sequence remains bounded (|z| ≤ 2), the point is in the set.

#### Step 4: Implementing the Kernel

```cpp
// Inside main(), after creating the image:

Kernel2D mandelbrot_kernel = [&](ImageFloat image) noexcept {
    // Get pixel coordinates
    UInt2 coord = dispatch_id().xy();
    Float2 uv = make_float2(coord) / make_float2(resolution);
    
    // Map to complex plane: centered at (-0.5, 0), zoomed in
    Float2 c = (uv - 0.5f) * 3.0f - make_float2(0.5f, 0.0f);
    
    // Mandelbrot iteration
    Float2 z = make_float2(0.0f);
    Float n = 0.0f;
    static constexpr int M = 100;  // Maximum iterations
    
    $for (i, M) {
        // z = z² + c
        Float2 z_new = make_float2(
            z.x * z.x - z.y * z.y + c.x,
            2.0f * z.x * z.y + c.y
        );
        z = z_new;
        
        // Check divergence
        $if (dot(z, z) > 4.0f) {
            $break;
        };
        n += 1.0f;
    };
    
    // Color using cosine palette (Inigo Quilez's technique)
    Float t = n / static_cast<float>(M);
    Float3 color = 0.5f + 0.5f * cos(6.28318f * (t + make_float3(0.0f, 0.33f, 0.67f)));
    
    // Write to image
    image.write(coord, make_float4(color, 1.0f));
};
```

#### Step 5: Compile and Execute

```cpp
// Compile the kernel
auto shader = device.compile(mandelbrot_kernel);

// Execute
stream << shader(image).dispatch(resolution)
       << synchronize();

// Save result
luisa::vector<uint8_t> pixels(resolution.x * resolution.y * 4);
stream << image.copy_to(pixels.data())
       << synchronize();

stbi_write_png("mandelbrot.png", resolution.x, resolution.y, 4, pixels.data(), 0);
LUISA_INFO("Saved to mandelbrot.png");
```

#### Complete Code

<details>
<summary>Click to expand complete code</summary>

```cpp
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    
    static constexpr uint2 resolution = make_uint2(1024u, 1024u);
    Image<float> image = device.create_image<float>(
        PixelStorage::FLOAT4, resolution.x, resolution.y);
    
    Kernel2D mandelbrot_kernel = [&](ImageFloat image) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float2 uv = make_float2(coord) / make_float2(resolution);
        
        Float2 c = (uv - 0.5f) * 3.0f - make_float2(0.5f, 0.0f);
        Float2 z = make_float2(0.0f);
        Float n = 0.0f;
        static constexpr int M = 100;
        
        $for (i, M) {
            Float2 z_new = make_float2(
                z.x * z.x - z.y * z.y + c.x,
                2.0f * z.x * z.y + c.y
            );
            z = z_new;
            $if (dot(z, z) > 4.0f) { $break; };
            n += 1.0f;
        };
        
        Float t = n / static_cast<float>(M);
        Float3 color = 0.5f + 0.5f * cos(6.28318f * (t + make_float3(0.0f, 0.33f, 0.67f)));
        image.write(coord, make_float4(color, 1.0f));
    };
    
    auto shader = device.compile(mandelbrot_kernel);
    stream << shader(image).dispatch(resolution) << synchronize();
    
    luisa::vector<uint8_t> pixels(resolution.x * resolution.y * 4);
    stream << image.copy_to(pixels.data()) << synchronize();
    stbi_write_png("mandelbrot.png", resolution.x, resolution.y, 4, pixels.data(), 0);
    
    return 0;
}
```
</details>

#### Exercises

1. **Zoom Animation**: Modify `c` to animate over time
2. **Julia Set**: Keep `c` constant and vary initial `z`
3. **Smooth Coloring**: Use `n - log2(log2(dot(z,z)))` for smoother gradients

---

## Path Tracing Renderer

Now let's build a more sophisticated renderer using ray tracing. We'll implement a basic path tracer for the Cornell Box scene.

### Final Result

A physically-based rendered image with global illumination.

### Step-by-Step Implementation

#### Step 1: Scene Setup

We need geometry (triangles), materials, and acceleration structures:

```cpp
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

// Cornell Box geometry data
static constexpr float3 vertices[] = {
    // Floor
    {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f},
    // Ceiling
    {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f},
    // ... more vertices
};

static constexpr uint3 triangles[] = {
    {0, 1, 2}, {0, 2, 3},  // Floor
    // ... more triangles
};
```

#### Step 2: Create Acceleration Structure

```cpp
// Create buffers
Buffer<float3> vertex_buffer = device.create_buffer<float3>(num_vertices);
Buffer<Triangle> triangle_buffer = device.create_buffer<Triangle>(num_triangles);

// Upload data
stream << vertex_buffer.copy_from(vertices)
       << triangle_buffer.copy_from(triangles)
       << synchronize();

// Create mesh and acceleration structure
Mesh mesh = device.create_mesh(vertex_buffer, triangle_buffer);
Accel accel = device.create_accel();
accel.emplace_back(mesh, make_float4x4(1.0f));  // Identity transform

stream << mesh.build() << accel.build() << synchronize();
```

#### Step 3: Helper Callables

```cpp
// Linear congruential generator for random numbers
Callable lcg = [](UInt &state) noexcept {
    constexpr uint a = 1664525u;
    constexpr uint c = 1013904223u;
    state = a * state + c;
    return cast<float>(state) / cast<float>(std::numeric_limits<uint>::max());
};

// Cosine-weighted hemisphere sampling
Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
    Float r = sqrt(u.x);
    Float theta = 2.0f * pi * u.y;
    return make_float3(r * cos(theta), r * sin(theta), sqrt(1.0f - u.x));
};

// Build orthonormal basis from normal
Callable make_onb = [](Float3 normal) noexcept {
    Float3 b1 = ite(abs(normal.y) < 0.999f, 
                     normalize(cross(normal, make_float3(0.0f, 1.0f, 0.0f))),
                     make_float3(1.0f, 0.0f, 0.0f));
    Float3 b2 = cross(normal, b1);
    return make_float3x3(b1, b2, normal);
};
```

#### Step 4: The Path Tracing Kernel

```cpp
Kernel2D path_tracer = [&](ImageFloat image, ImageUInt seed_image, 
                           AccelVar accel, UInt frame_index) noexcept {
    set_block_size(16u, 16u, 1u);
    
    UInt2 coord = dispatch_id().xy();
    Float2 resolution = make_float2(dispatch_size().xy());
    
    // Initialize random seed
    $if (frame_index == 0u) {
        seed_image.write(coord, make_uint4(coord.x * 1234567u + coord.y));
    };
    
    UInt seed = seed_image.read(coord).x;
    
    // Generate camera ray
    Float2 uv = (make_float2(coord) + make_float2(lcg(seed), lcg(seed))) / resolution;
    Float2 ndc = uv * 2.0f - 1.0f;
    
    Float3 origin = make_float3(0.0f, 0.0f, 3.0f);
    Float3 direction = normalize(make_float3(ndc.x, ndc.y, -1.0f));
    
    // Path tracing loop
    Float3 radiance = def(make_float3(0.0f));
    Float3 throughput = def(make_float3(1.0f));
    
    $for (bounce, 5u) {  // Maximum 5 bounces
        // Trace ray
        Var<Ray> ray = make_ray(origin, direction);
        Var<TriangleHit> hit = accel.intersect(ray, {});
        
        $if (hit->miss()) { $break; };
        
        // Get hit information
        Float3 p0 = vertex_buffer->read(hit.prim * 3u + 0u);
        Float3 p1 = vertex_buffer->read(hit.prim * 3u + 1u);
        Float3 p2 = vertex_buffer->read(hit.prim * 3u + 2u);
        Float3 p = hit->interpolate(p0, p1, p2);
        Float3 n = normalize(cross(p1 - p0, p2 - p0));
        
        // Simple diffuse material (white)
        Float3 albedo = make_float3(0.8f);
        
        // Add emission if hit light
        $if (hit.prim >= light_triangle_start) {
            radiance += throughput * make_float3(10.0f);  // Emissive
            $break;
        };
        
        // Sample next direction (cosine-weighted)
        Float2 u = make_float2(lcg(seed), lcg(seed));
        Float3 local_dir = cosine_sample_hemisphere(u);
        Var<Float3x3> onb = make_onb(n);
        Float3 new_dir = onb * local_dir;
        
        // Update throughput
        Float cos_theta = dot(new_dir, n);
        throughput *= albedo * cos_theta * (1.0f / pi);
        
        // Russian roulette
        Float p_survive = max(throughput.x, max(throughput.y, throughput.z));
        $if (lcg(seed) > p_survive) { $break; };
        throughput *= 1.0f / p_survive;
        
        // Setup next ray
        origin = p + n * 1e-4f;
        direction = new_dir;
    };
    
    // Accumulate over frames
    Float4 prev = image.read(coord);
    Float count = prev.w + 1.0f;
    Float3 avg = lerp(prev.xyz(), radiance, 1.0f / count);
    image.write(coord, make_float4(avg, count));
    
    seed_image.write(coord, make_uint4(seed));
};
```

#### Step 5: Rendering Loop

```cpp
// Create images
Image<float> framebuffer = device.create_image<float>(
    PixelStorage::FLOAT4, resolution.x, resolution.y);
Image<uint> seed_image = device.create_image<uint>(
    PixelStorage::INT1, resolution.x, resolution.y);

auto renderer = device.compile(path_tracer);

// Render multiple frames for accumulation
for (uint frame = 0; frame < 100; frame++) {
    stream << renderer(framebuffer, seed_image, accel, frame).dispatch(resolution)
           << synchronize();
}

// Tonemap and save
Kernel2D tonemap = [&](ImageFloat hdr, ImageFloat ldr) noexcept {
    UInt2 coord = dispatch_id().xy();
    Float3 color = hdr.read(coord).xyz();
    // Simple Reinhard tonemapping
    color = color / (color + 1.0f);
    // Gamma correction
    color = pow(color, 1.0f / 2.2f);
    ldr.write(coord, make_float4(color, 1.0f));
};

Image<float> ldr = device.create_image<float>(
    PixelStorage::BYTE4, resolution.x, resolution.y);
stream << device.compile(tonemap)(framebuffer, ldr).dispatch(resolution)
       << synchronize();

// Save...
```

#### Key Concepts Explained

1. **Path Tracing**: Simulates light transport by tracing random paths from camera
2. **Russian Roulette**: Probabilistically terminates paths to avoid infinite recursion
3. **Cosine Sampling**: Importance sampling for diffuse surfaces
4. **Accumulation**: Multiple samples averaged for noise reduction

---

## MPM Fluid Simulation

Material Point Method (MPM) is a hybrid Lagrangian-Eulerian method for simulating continua. We'll implement a basic 3D fluid simulation.

### Final Result

An interactive 3D fluid simulation with particles.

### Step-by-Step Implementation

#### Step 1: Simulation Parameters

```cpp
static constexpr int n_grid = 64;           // Grid resolution
static constexpr uint n_particles = 50000;  // Number of particles
static constexpr float dx = 1.0f / n_grid;  // Grid spacing
static constexpr float dt = 8e-5f;          // Time step
static constexpr float p_vol = (dx * 0.5f) * (dx * 0.5f) * (dx * 0.5f);
static constexpr float p_mass = 1.0f * p_vol;
static constexpr float E = 400.0f;          // Young's modulus
static constexpr float gravity = 9.8f;
```

#### Step 2: Data Structures

```cpp
// Particle data (Lagrangian)
Buffer<float3> x = device.create_buffer<float3>(n_particles);  // Positions
Buffer<float3> v = device.create_buffer<float3>(n_particles);  // Velocities
Buffer<float3x3> C = device.create_buffer<float3x3>(n_particles);  // Affine momentum
Buffer<float> J = device.create_buffer<float>(n_particles);    // Volume ratio

// Grid data (Eulerian)
Buffer<float4> grid = device.create_buffer<float4>(n_grid * n_grid * n_grid);
// xyz = momentum, w = mass
```

#### Step 3: Grid Clearing Kernel

```cpp
auto clear_grid = device.compile<3>([&] {
    set_block_size(8, 8, 1);
    UInt idx = dispatch_id().x + dispatch_id().y * n_grid 
               + dispatch_id().z * n_grid * n_grid;
    grid->write(idx, make_float4(0.0f));
});
```

#### Step 4: Particle-to-Grid Transfer

This is the heart of MPM - transferring particle data to the grid:

```cpp
auto particle_to_grid = device.compile<1>([&] {
    set_block_size(64, 1, 1);
    UInt p = dispatch_id().x;
    
    // Particle position in grid coordinates
    Float3 Xp = x->read(p) / dx;
    Int3 base = make_int3(Xp - 0.5f);
    Float3 fx = Xp - make_float3(base);
    
    // Quadratic B-spline weights
    auto w = [&](Float3 fx) {
        return make_float3(
            0.5f * (1.5f - fx) * (1.5f - fx),
            0.75f - (fx - 1.0f) * (fx - 1.0f),
            0.5f * (fx - 0.5f) * (fx - 0.5f)
        );
    };
    Float3 wx = w(fx.x), wy = w(fx.y), wz = w(fx.z);
    
    // Stress-based force (simplified for fluids)
    Float stress = -4.0f * dt * E * p_vol * (J->read(p) - 1.0f) / (dx * dx);
    Float3x3 affine = make_float3x3(stress, 0.0f, 0.0f,
                                     0.0f, stress, 0.0f,
                                     0.0f, 0.0f, stress);
    Float3 vp = v->read(p);
    
    // Scatter to 3x3x3 grid nodes
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                Float3 offset = make_float3(i, j, k);
                Float3 dpos = (offset - fx) * dx;
                Float weight = wx[i] * wy[j] * wz[k];
                
                Float3 v_add = weight * (p_mass * vp + affine * dpos);
                UInt idx = (base.x + i) + (base.y + j) * n_grid 
                          + (base.z + k) * n_grid * n_grid;
                
                // Atomic add to grid
                grid->atomic(idx).x.fetch_add(v_add.x);
                grid->atomic(idx).y.fetch_add(v_add.y);
                grid->atomic(idx).z.fetch_add(v_add.z);
                grid->atomic(idx).w.fetch_add(weight * p_mass);
            }
        }
    }
});
```

#### Step 5: Grid Velocity Update

```cpp
auto grid_update = device.compile<3>([&] {
    set_block_size(8, 8, 1);
    Int3 coord = make_int3(dispatch_id().xyz());
    
    UInt idx = coord.x + coord.y * n_grid + coord.z * n_grid * n_grid;
    Float4 v_and_m = grid->read(idx);
    
    Float3 v = v_and_m.xyz();
    Float m = v_and_m.w;
    
    // Normalize by mass
    v = ite(m > 0.0f, v / m, v);
    
    // Gravity
    v.y -= dt * gravity;
    
    // Boundary conditions
    static constexpr int bound = 3;
    v = ite((coord < bound && v < 0.0f) || 
            (coord > n_grid - bound && v > 0.0f), 
            0.0f, v);
    
    grid->write(idx, make_float4(v, m));
});
```

#### Step 6: Grid-to-Particle Transfer

```cpp
auto grid_to_particle = device.compile<1>([&] {
    set_block_size(64, 1, 1);
    UInt p = dispatch_id().x;
    
    Float3 Xp = x->read(p) / dx;
    Int3 base = make_int3(Xp - 0.5f);
    Float3 fx = Xp - make_float3(base);
    
    // Same weights as P2G
    // ... weight computation ...
    
    Float3 new_v = def(make_float3(0.0f));
    Float3x3 new_C = def(make_float3x3(0.0f));
    
    // Gather from grid
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                Float3 offset = make_float3(i, j, k);
                Float3 dpos = (offset - fx) * dx;
                Float weight = wx[i] * wy[j] * wz[k];
                
                UInt idx = (base.x + i) + (base.y + j) * n_grid 
                          + (base.z + k) * n_grid * n_grid;
                Float3 g_v = grid->read(idx).xyz();
                
                new_v += weight * g_v;
                new_C += 4.0f * weight * outer_product(g_v, dpos) / (dx * dx);
            }
        }
    }
    
    // Update particle state
    v->write(p, new_v);
    x->write(p, x->read(p) + new_v * dt);
    J->write(p, J->read(p) * (1.0f + dt * trace(new_C)));
    C->write(p, new_C);
});
```

#### Step 7: Simulation Loop

```cpp
// Initialize particles
luisa::vector<float3> x_init(n_particles);
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(0.2f, 0.8f);

for (uint i = 0; i < n_particles; i++) {
    x_init[i] = make_float3(dis(gen), dis(gen), dis(gen));
}

stream << x.copy_from(x_init.data())
       << v.copy_from(luisa::vector<float3>(n_particles, make_float3(0.0f)).data())
       << J.copy_from(luisa::vector<float>(n_particles, 1.0f).data())
       << synchronize();

// Simulation loop
Window window{"MPM Fluid", 1024, 1024};
Swapchain swapchain = device.create_swapchain(stream, {
    .display = window.native_display(),
    .window = window.native_handle(),
    .size = make_uint2(1024u, 1024u),
});
Image<float> display = device.create_image<float>(
    swapchain.backend_storage(), make_uint2(1024u, 1024u));

while (!window.should_close()) {
    CommandList cmd;
    
    // MPM sub-steps
    for (int i = 0; i < 25; i++) {
        cmd << clear_grid().dispatch(n_grid, n_grid, n_grid)
            << particle_to_grid().dispatch(n_particles)
            << grid_update().dispatch(n_grid, n_grid, n_grid)
            << grid_to_particle().dispatch(n_particles);
    }
    
    // Render particles to display
    cmd << clear_display().dispatch(1024, 1024)
        << draw_particles().dispatch(n_particles)
        << swapchain.present(display);
    
    stream << cmd.commit();
    window.poll_events();
}
```

#### Key Concepts Explained

1. **Lagrangian vs Eulerian**: Particles track material; grid handles physics
2. **APIC (Affine Particle-in-Cell)**: Uses affine momentum matrix for better accuracy
3. **Transfer**: P2G (particle to grid) and G2P (grid to particle) are the core operations
4. **MLS-MPM**: Moving Least Squares MPM for stability

---

## Summary

These tutorials demonstrate:

1. **Mandelbrot**: Basic kernel structure, loops, and image output
2. **Path Tracer**: Ray tracing, acceleration structures, sampling
3. **MPM**: Complex simulation with multiple kernels, atomic operations

### Common Patterns

All examples follow this structure:

```cpp
// 1. Setup
Context context{argv[0]};
Device device = context.create_device("cuda");
Stream stream = device.create_stream();

// 2. Create resources
Buffer<T> buffer = device.create_buffer<T>(size);
Image<float> image = device.create_image<float>(...);

// 3. Define kernels
Kernel2D kernel = [&](...) noexcept {
    // GPU code
};

// 4. Compile and execute
auto shader = device.compile(kernel);
stream << shader(args...).dispatch(size)
       << synchronize();

// 5. Retrieve results
stream << buffer.copy_to(host_data.data())
       << synchronize();
```

---

## Conway's Game of Life

Cellular automata are fascinating discrete systems. Let's implement Conway's Game of Life, a classic example that demonstrates **ping-pong buffering** - a technique where we alternate between two buffers to avoid read-write conflicts.

### Final Result

An interactive simulation where cells evolve based on simple rules, creating complex emergent patterns.

### Step-by-Step Implementation

#### Step 1: Understanding the Rules

Conway's Game of Life follows three simple rules:
1. **Survival**: Live cells with 2-3 neighbors survive
2. **Birth**: Dead cells with exactly 3 neighbors become alive
3. **Death**: All other cells die or stay dead

#### Step 2: The ImagePair Structure

We'll use two images that swap each frame:

```cpp
struct ImagePair {
    Image<uint> prev;
    Image<uint> curr;
    ImagePair(Device &device, PixelStorage storage, uint width, uint height) noexcept
        : prev{device.create_image<uint>(storage, width, height)},
          curr{device.create_image<uint>(storage, width, height)} {}
    void swap() noexcept { std::swap(prev, curr); }
};
```

#### Step 3: The Update Kernel

```cpp
// Helper to read cell state
Callable read_state = [](ImageUInt prev, UInt2 uv) noexcept {
    return prev.read(uv).x == 255u;
};

Kernel2D kernel = [&](ImageUInt prev, ImageUInt curr) noexcept {
    set_block_size(16, 16, 1);
    UInt count = def(0u);
    UInt2 uv = dispatch_id().xy();
    UInt2 size = dispatch_size().xy();
    Bool state = read_state(prev, uv);
    Int2 p = make_int2(uv);
    
    // Check all 8 neighbors with toroidal wrapping
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                Int2 q = p + make_int2(dx, dy) + make_int2(size);
                Bool neighbor = read_state(prev, make_uint2(q) % size);
                count += ite(neighbor, 1, 0);
            }
        }
    }
    
    // Apply Conway's rules
    Bool c0 = count == 2u;
    Bool c1 = count == 3u;
    curr.write(uv, make_uint4(make_uint3(ite((state & c0) | c1, 255u, 0u)), 255u));
};
```

#### Step 4: Display Kernel

We upscale the low-res simulation for better visibility:

```cpp
Kernel2D display_kernel = [&](ImageUInt in_tex, ImageFloat out_tex) noexcept {
    set_block_size(16, 16, 1);
    UInt2 uv = dispatch_id().xy();
    UInt2 coord = uv / 4u;  // 4x upscaling
    UInt4 value = in_tex.read(coord);
    out_tex.write(uv, make_float4(value) / 255.0f);
};
```

#### Step 5: Initialization and Main Loop

```cpp
// Initialize with random state (25% alive)
luisa::vector<uint> host_image(width * height);
for (auto &v : host_image) {
    auto x = (rng() % 4u == 0u) * 255u;
    v = x * 0x00010101u | 0xff000000u;  // White or black
}
stream << image_pair.prev.copy_from(host_image.data()) << synchronize();

// Main loop
while (!window.should_close()) {
    stream << shader(image_pair.prev, image_pair.curr).dispatch(width, height)
           << display_shader(image_pair.curr, display).dispatch(width * 4u, height * 4u)
           << swap_chain.present(display);
    image_pair.swap();  // Ping-pong: swap buffers
    std::this_thread::sleep_for(std::chrono::milliseconds{30});
    window.poll_events();
}
```

#### Key Concepts

1. **Ping-Pong Buffering**: We read from `prev` and write to `curr`, then swap
2. **Toroidal Boundary**: `% size` makes the grid wrap around (edges connect)
3. **Neighborhood**: The 8 surrounding cells determine each cell's fate

#### Exercises

1. **Different Rules**: Try HighLife (born with 3 or 6 neighbors)
2. **Pattern Library**: Implement a pattern loader for gliders, blinkers, etc.
3. **Multi-State**: Add fading effects for recently-dead cells

---

## Wave Equation Simulation

Simulate realistic water ripples using the **finite difference method** to solve the 2D wave equation. This demonstrates PDE solving and interactive GPU computing.

### Final Result

An interactive water simulation where you can drop droplets and watch ripples propagate with realistic physics.

### The Physics

The wave equation: `∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)`

Discretized form: `u_next = 2*u_curr - u_prev + c²*dt²*laplacian(u)`

### Step-by-Step Implementation

#### Step 1: Setup Three Buffers

We need three height buffers for time integration:

```cpp
Image<float> height_prev = device.create_image<float>(PixelStorage::FLOAT1, width, height);
Image<float> height_curr = device.create_image<float>(PixelStorage::FLOAT1, width, height);
Image<float> height_next = device.create_image<float>(PixelStorage::FLOAT1, width, height);
```

#### Step 2: The Wave Solver Kernel

```cpp
Kernel2D wave_step = [&](ImageFloat prev, ImageFloat curr, ImageFloat next) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();
    Var size = dispatch_size().xy();

    // Sample neighbors with wrapping
    auto sample = [&](Int2 offset) noexcept {
        Var sample_uv = make_uint2(
            (cast<int>(uv.x) + offset.x + cast<int>(size.x)) % cast<int>(size.x),
            (cast<int>(uv.y) + offset.y + cast<int>(size.y)) % cast<int>(size.y));
        return curr.read(sample_uv).x;
    };

    Var u_curr = curr.read(uv).x;
    Var u_left = sample(make_int2(-1, 0));
    Var u_right = sample(make_int2(1, 0));
    Var u_up = sample(make_int2(0, -1));
    Var u_down = sample(make_int2(0, 1));

    // Laplacian: sum of neighbors - 4*center
    Var laplacian = u_left + u_right + u_up + u_down - 4.0f * u_curr;

    // Wave equation
    Var u_prev = prev.read(uv).x;
    Var u_next = 2.0f * u_curr - u_prev + c * c * dt * dt * laplacian;
    
    // Damping prevents infinite oscillation
    u_next *= damping;
    u_next = clamp(u_next, -10.0f, 10.0f);

    next.write(uv, make_float4(u_next, 0.0f, 0.0f, 1.0f));
};
```

#### Step 3: Interactive Droplet Drop

```cpp
Kernel2D drop_droplet = [](ImageFloat height, UInt2 center, Float strength) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();

    // Gaussian drop shape
    Var dx = cast<float>(cast<int>(uv.x) - cast<int>(center.x));
    Var dy = cast<float>(cast<int>(uv.y) - cast<int>(center.y));
    Var dist_sq = dx * dx + dy * dy;
    
    Var radius = 20.0f;
    Var gaussian = exp(-dist_sq / (radius * radius * 0.5f));

    Var current = height.read(uv).x;
    height.write(uv, make_float4(current + strength * gaussian, 0.0f, 0.0f, 1.0f));
};
```

#### Step 4: Rendering with Normal Mapping

```cpp
Kernel2D render_kernel = [](ImageFloat height, ImageFloat output, Float3 light_dir) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();
    
    // Compute normal from height gradient
    Var h_center = height.read(uv).x;
    Var h_left = sample_height(make_int2(-1, 0));
    Var h_right = sample_height(make_int2(1, 0));
    Var h_up = sample_height(make_int2(0, -1));
    Var h_down = sample_height(make_int2(0, 1));

    Var normal = normalize(make_float3(
        (h_left - h_right) * 0.5f,
        (h_up - h_down) * 0.5f,
        1.0f));

    // Blinn-Phong lighting
    Var N = normal;
    Var L = normalize(light_dir);
    Var V = make_float3(0.0f, 0.0f, 1.0f);
    Var H = normalize(L + V);
    
    Var diff = max(dot(N, L), 0.0f);
    Var spec = pow(max(dot(N, H), 0.0f), 64.0f);

    // Water color with foam at peaks
    Var water_color = make_float3(0.0f, 0.3f, 0.5f);
    Var foam_color = make_float3(0.9f, 0.95f, 1.0f);
    Var foam = smoothstep(0.3f, 0.8f, h_center);
    Var base_color = lerp(water_color, foam_color, foam);
    
    Var lit_color = base_color * (0.3f + 0.7f * diff) + make_float3(spec * 0.5f);
    output.write(uv, make_float4(lit_color, 1.0f));
};
```

#### Step 5: Main Simulation Loop

```cpp
while (!window.should_close()) {
    // Handle mouse input for droplet drop
    if (window.is_mouse_down(MouseButton::LEFT)) {
        uint sim_x = mouse_pos.x / display_scale;
        uint sim_y = mouse_pos.y / display_scale;
        stream << drop_shader(height_curr, make_uint2(sim_x, sim_y), -1.0f)
                      .dispatch(width, height);
    }

    // Auto rain occasionally
    if (rng() % 60u == 0u) {
        stream << drop_shader(height_curr, random_pos, -0.3f).dispatch(width, height);
    }

    // Update wave simulation
    stream << wave_shader(height_prev, height_curr, height_next).dispatch(width, height);
    
    // Swap buffers: prev <- curr, curr <- next
    std::swap(height_prev, height_curr);
    std::swap(height_curr, height_next);

    // Render with rotating light
    rot_time += 0.02f;
    Float3 light_dir{cosf(rot_time), sinf(rot_time), 0.8f};
    stream << render(height_curr, temp_display, light_dir).dispatch(width, height)
           << upscale(temp_display, display).dispatch(display_width, display_height)
           << swap_chain.present(display);
}
```

#### Key Concepts

1. **Finite Differences**: Approximate derivatives using neighbor samples
2. **Three-Buffer Scheme**: Need past, present, and future for second-order time integration
3. **Normal Mapping**: Create realistic lighting from height field gradients

#### Exercises

1. **Different Boundary Conditions**: Try clamped or reflective boundaries
2. **Variable Speed**: Make wave speed vary by position for interesting effects
3. **Multiple Drops**: Implement a buffer of pending droplets

---

## Image Processing Pipeline

Build a multi-stage image processing pipeline demonstrating **separable filters**, **convolution**, and **multi-pass rendering**.

### Final Result

A real-time image processing demo showing Gaussian blur, Sobel edge detection, and creative compositing.

### Step-by-Step Implementation

#### Step 1: Generate Test Pattern

```cpp
Kernel2D generate_pattern = [](ImageFloat image) noexcept {
    set_block_size(16, 16, 1);
    Var uv = make_float2(dispatch_id().xy()) / make_float2(dispatch_size().xy());

    // Checkerboard
    Var checker = sin(uv.x * 20.0f) * sin(uv.y * 20.0f);
    
    // Concentric circles
    Var center = uv - make_float2(0.5f);
    Var radius = length(center);
    Var circles = sin(radius * 50.0f);
    
    // Combine patterns
    Var pattern = checker * 0.3f + circles * 0.4f;
    
    // Add "face-like" features
    Var eye1 = exp(-length(uv - make_float2(0.35f, 0.4f)) * 50.0f);
    Var eye2 = exp(-length(uv - make_float2(0.65f, 0.4f)) * 50.0f);
    Var mouth = exp(-pow(uv.y - 0.6f - 0.1f * pow(uv.x - 0.5f, 2.0f), 2.0f) * 200.0f);

    Var final_val = pattern * 0.5f + 0.5f + eye1 * 0.3f + eye2 * 0.3f + mouth * 0.2f;
    image.write(dispatch_id().xy(), make_float4(make_float3(final_val), 1.0f));
};
```

#### Step 2: Separable Gaussian Blur

Instead of a 2D convolution (81 samples), we do two 1D passes (18 samples total):

```cpp
// Horizontal pass
Kernel2D gaussian_blur_h = [](ImageFloat input, ImageFloat output) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();
    Var size = dispatch_size().xy();

    // Gaussian weights (sigma=2.0)
    Var weights = make_float9(0.006f, 0.028f, 0.084f, 0.168f, 0.224f,
                              0.168f, 0.084f, 0.028f, 0.006f);
    Var offsets = make_int9(-4, -3, -2, -1, 0, 1, 2, 3, 4);

    Var sum = make_float3(0.0f);
    for (uint i = 0u; i < 9u; i++) {
        Var sample_uv = make_uint2(
            clamp(cast<int>(uv.x) + offsets[i], 0, cast<int>(size.x) - 1),
            uv.y);
        sum += input.read(sample_uv).xyz() * weights[i];
    }
    output.write(uv, make_float4(sum, 1.0f));
};

// Vertical pass (similar structure)
Kernel2D gaussian_blur_v = [](ImageFloat input, ImageFloat output) noexcept { ... };
```

#### Step 3: Sobel Edge Detection

```cpp
Kernel2D sobel_edge = [](ImageFloat input, ImageFloat output) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();
    Var size = dispatch_size().xy();

    auto sample = [&](Int2 offset) noexcept {
        Var sample_uv = make_uint2(
            clamp(cast<int>(uv.x) + offset.x, 0, cast<int>(size.x) - 1),
            clamp(cast<int>(uv.y) + offset.y, 0, cast<int>(size.y) - 1));
        return input.read(sample_uv).x;  // Luminance
    };

    // Sobel X kernel: [-1 0 1; -2 0 2; -1 0 1]
    Var gx = sample(make_int2(-1, -1)) * -1.0f +
             sample(make_int2(1, -1)) * 1.0f +
             sample(make_int2(-1, 0)) * -2.0f +
             sample(make_int2(1, 0)) * 2.0f +
             sample(make_int2(-1, 1)) * -1.0f +
             sample(make_int2(1, 1)) * 1.0f;

    // Sobel Y kernel: [-1 -2 -1; 0 0 0; 1 2 1]
    Var gy = sample(make_int2(-1, -1)) * -1.0f +
             sample(make_int2(0, -1)) * -2.0f +
             sample(make_int2(1, -1)) * -1.0f +
             sample(make_int2(-1, 1)) * 1.0f +
             sample(make_int2(0, 1)) * 2.0f +
             sample(make_int2(1, 1)) * 1.0f;

    // Gradient magnitude and direction
    Var magnitude = sqrt(gx * gx + gy * gy);
    Var angle = atan2(gy, gx) / 3.14159f * 0.5f + 0.5f;
    
    output.write(uv, make_float4(magnitude, angle, 0.0f, 1.0f));
};
```

#### Step 4: Creative Compositing

```cpp
Kernel2D composite = [](ImageFloat original, ImageFloat blurred,
                       ImageFloat edges, ImageFloat output, Float edge_intensity) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();

    Var orig = original.read(uv);
    Var blur = blurred.read(uv);
    Var edge = edges.read(uv);

    // Unsharp mask sharpening
    Var sharpen_amount = 1.5f;
    Var sharpened = orig.xyz() + (orig.xyz() - blur.xyz()) * sharpen_amount;
    sharpened = clamp(sharpened, 0.0f, 1.0f);

    // Edge overlay with cyan color
    Var edge_color = make_float3(0.0f, 0.8f, 1.0f) * edge.x * edge_intensity;
    Var final_color = lerp(sharpened, edge_color, edge.x * 0.5f);

    // Vignette effect
    Var coord = make_float2(uv) / make_float2(dispatch_size().xy());
    Var center_dist = length(coord - make_float2(0.5f));
    Var vignette = 1.0f - center_dist * 0.5f;

    output.write(uv, make_float4(final_color * vignette, 1.0f));
};
```

#### Step 5: Pipeline Execution

```cpp
// Create intermediate images
Image<float> source_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
Image<float> temp_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
Image<float> blurred_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
Image<float> edge_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);

// Generate source
stream << pattern_shader(source_image).dispatch(width, height);

// Run pipeline once
stream << blur_h_shader(source_image, temp_image).dispatch(width, height)
       << blur_v_shader(temp_image, blurred_image).dispatch(width, height)
       << sobel_shader(source_image, edge_image).dispatch(width, height);

// Animate in main loop
while (!window.should_close()) {
    float edge_intensity = (sinf(time) + 1.0f) * 0.5f;  // Pulsing effect
    stream << composite_shader(source_image, blurred_image, edge_image,
                               display, edge_intensity).dispatch(width, height)
           << swap_chain.present(display);
}
```

#### Key Concepts

1. **Separable Filters**: Decompose 2D convolution into two 1D passes for efficiency
2. **Multi-Pass Rendering**: Chain multiple kernels for complex effects
3. **Unsharp Mask**: Sharpening using `original + (original - blurred) * amount`

#### Exercises

1. **Bloom Effect**: Add a bright-pass filter before blur
2. **Custom Kernels**: Implement emboss or motion blur filters
3. **Live Input**: Use camera input instead of procedural pattern

---

## N-Body Gravitational Simulation

Simulate thousands of stars interacting through gravity. This demonstrates **particle systems**, **tile-based optimization**, and **3D rendering** on the GPU.

### Final Result

A rotating galaxy simulation with thousands of gravitationally interacting particles.

### Step-by-Step Implementation

#### Step 1: Particle Structure

```cpp
struct Particle {
    float3 position;
    float3 velocity;
    float mass;
    float pad[3];  // Alignment padding
};

LUISA_STRUCT(Particle, position, velocity, mass, pad) {};
```

#### Step 2: Galaxy Initialization

```cpp
// Initialize in a disk-like galaxy configuration
luisa::vector<Particle> host_particles(n_particles);
for (uint i = 0u; i < n_particles; i++) {
    float radius = dist_radius(rng);
    float angle = dist_angle(rng);
    float height = (rng() / float(UINT32_MAX) - 0.5f) * 0.1f;

    // Position in disk
    float3 pos{
        radius * cosf(angle),
        height,
        radius * sinf(angle)};

    // Tangential velocity for orbital motion
    float orbital_speed = sqrtf(G * 1000.0f / radius);
    float3 vel{
        -orbital_speed * sinf(angle),
        0.0f,
        orbital_speed * cosf(angle)};

    host_particles[i] = Particle{
        .position = pos,
        .velocity = vel,
        .mass = dist_mass(rng),
        .pad = {0.0f, 0.0f, 0.0f}};
}
stream << particles_read.copy_from(host_particles.data()) << synchronize();
```

#### Step 3: N-Body Physics Kernel

```cpp
Kernel1D nbody_kernel = [&](BufferVar<Particle> read_buf, 
                            BufferVar<Particle> write_buf) noexcept {
    set_block_size(tile_size);
    Var idx = dispatch_id().x;
    Var p = read_buf.read(idx);

    // Accumulate gravitational forces
    Var force = make_float3(0.0f);

    // Tile loop - process particles in chunks
    for (uint tile = 0u; tile < n_particles; tile += tile_size) {
        for (uint j = 0u; j < tile_size; j++) {
            Var other_idx = tile + j;
            
            // Skip self-interaction to avoid infinity
            if_(other_idx != idx, [&] {
                Var other = read_buf.read(other_idx);
                Var r = other.position - p.position;
                Var dist_sq = dot(r, r) + softening * softening;
                Var dist = sqrt(dist_sq);
                Var f = G * p.mass * other.mass / dist_sq;
                force += f * r / dist;  // F = G*m1*m2/r² * direction
            });
        }
    }

    // Euler integration
    Var new_vel = p.velocity + force / p.mass * dt;
    Var new_pos = p.position + new_vel * dt;

    write_buf.write(idx, Particle{
        .position = new_pos,
        .velocity = new_vel,
        .mass = p.mass,
        .pad = {0.0f, 0.0f, 0.0f}});
};
```

#### Step 4: 3D Rendering Kernel

```cpp
Kernel2D render_kernel = [&](BufferVar<Particle> particles, 
                             ImageFloat image, Float rot_x, Float rot_y) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();
    Var size = dispatch_size().xy();

    // Clear background
    image.write(uv, make_float4(0.02f, 0.02f, 0.05f, 1.0f));

    Var color = make_float3(0.0f);

    // Project particles
    for (uint i = 0u; i < n_particles; i += 64u) {
        Var p = particles.read(i);

        // Rotation around Y axis
        Var cos_y = cos(rot_y);
        Var sin_y = sin(rot_y);
        Var x1 = p.position.x * cos_y - p.position.z * sin_y;
        Var z1 = p.position.x * sin_y + p.position.z * cos_y;
        Var y1 = p.position.y;

        // Rotation around X axis
        Var cos_x = cos(rot_x);
        Var sin_x = sin(rot_x);
        Var y2 = y1 * cos_x - z1 * sin_x;
        Var z2 = y1 * sin_x + z1 * cos_x;

        // Perspective projection
        Var distance = 5.0f + z2;
        Var scale = 2.0f / distance;
        Var screen_x = (x1 * scale + 1.0f) * 0.5f * cast<float>(size.x);
        Var screen_y = (y2 * scale + 1.0f) * 0.5f * cast<float>(size.y);

        // Gaussian glow
        Var dx = cast<float>(uv.x) - screen_x;
        Var dy = cast<float>(uv.y) - screen_y;
        Var d_sq = dx * dx + dy * dy;
        Var intensity = exp(-d_sq / (50.0f * scale));

        // Color based on particle index
        Var particle_color = make_float3(
            0.8f + 0.2f * sin(cast<float>(i)),
            0.6f + 0.3f * cos(cast<float>(i) * 1.3f),
            1.0f);

        color += particle_color * intensity * 0.1f;
    }

    // Motion blur effect
    Var old = image.read(uv).xyz();
    Var final_color = lerp(old, min(color, 1.0f), 0.3f);
    image.write(uv, make_float4(final_color, 1.0f));
};
```

#### Step 5: Main Loop

```cpp
while (!window.should_close()) {
    // Auto-rotate
    rot_y += 0.002f;

    // Physics update (double buffering)
    stream << nbody_shader(particles_read, particles_write).dispatch(n_particles);
    std::swap(particles_read, particles_write);

    // Render every few frames
    if (frame % 3u == 0u) {
        stream << render_shader(particles_read, display, rot_x, rot_y)
                      .dispatch(width, height)
               << swap_chain.present(display);
    }
    frame++;
}
```

#### Key Concepts

1. **Softening**: Add small epsilon to prevent division by zero at close range
2. **Tile-Based Processing**: Process in chunks for cache efficiency
3. **Double Buffering**: Read from one buffer, write to another, then swap
4. **Gaussian Splatting**: Render particles as soft glows rather than points

#### Exercises

1. **Barnes-Hut**: Implement O(n log n) hierarchical approximation
2. **Collision Detection**: Add particle collisions
3. **Different Forces**: Try Coulomb (electrostatic) or Lennard-Jones (molecular)

---

## Summary

These seven tutorials demonstrate:

1. **Mandelbrot**: Basic kernel structure, loops, and image output
2. **Path Tracer**: Ray tracing, acceleration structures, sampling
3. **MPM**: Complex simulation with multiple kernels, atomic operations
4. **Game of Life**: Cellular automata, ping-pong buffering
5. **Wave Equation**: PDE solving, interactive simulation
6. **Image Processing**: Multi-pass pipelines, convolution
7. **N-Body**: Particle systems, tile-based optimization

### Performance Tips

1. **Use appropriate block sizes**: 16×16 for 2D kernels, 256 for 1D
2. **Minimize memory transfers**: Keep data on GPU when possible
3. **Batch operations**: Use `CommandList` for multiple dispatches
4. **Profile with tools**: Use Nsight, RenderDoc, or vendor tools

### Next Steps

- Explore the [LuisaCompute tests](https://github.com/LuisaGroup/LuisaCompute/tree/stable/src/tests) for more examples
- Check [LuisaRender](https://github.com/LuisaGroup/LuisaRender) for production rendering
- Try implementing your own simulations!
