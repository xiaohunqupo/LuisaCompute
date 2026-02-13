# Tutorials

This section provides step-by-step tutorials for building practical applications with LuisaCompute.

## Table of Contents

1. [ShaderToy-Style Mandelbrot](#shadertoy-style-mandelbrot) - A gentle introduction
2. [Path Tracing Renderer](#path-tracing-renderer) - Global illumination with ray tracing
3. [MPM Fluid Simulation](#mpm-fluid-simulation) - Material Point Method for fluids

---

## ShaderToy-Style Mandelbrot

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

### Next Steps

- Explore the [LuisaCompute tests](../../src/tests) for more examples
- Check [LuisaRender](https://github.com/LuisaGroup/LuisaRender) for production rendering
- Try implementing your own simulations!
