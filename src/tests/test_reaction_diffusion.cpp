// Reaction-Diffusion Simulation (Gray-Scott Model)
// Simulates pattern formation using the Gray-Scott reaction-diffusion equations.
// This mathematical model demonstrates how complex patterns can emerge from
// simple chemical reactions and diffusion processes.
//
// The Gray-Scott equations:
//   du/dt = Du * laplacian(u) - u*v^2 + F*(1-u)
//   dv/dt = Dv * laplacian(v) + u*v^2 - (F+k)*v
//
// Where:
//   u = concentration of chemical A (inhibitor)
//   v = concentration of chemical B (activator)
//   Du, Dv = diffusion rates
//   F = feed rate
//   k = kill rate
//
// Features demonstrated:
// - Solving coupled PDEs on the GPU
// - Ping-pong buffering for time evolution
// - Parameter exploration for different pattern types
// - Real-time visualization with color mapping

#include <random>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/sugar.h>
#include <luisa/gui/window.h>

using namespace luisa;
using namespace luisa::compute;

// Chemical concentration pair
struct ChemicalState {
    float u;  // Inhibitor concentration
    float v;  // Activator concentration
};

LUISA_STRUCT(ChemicalState, u, v) {};

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    LUISA_INFO("Gray-Scott Reaction-Diffusion Simulation");
    LUISA_INFO("Controls: 1=Coral, 2=Fingerprint, 3=Spots, 4=Stripes, R=Reset, SPACE=Pause");

    // Simulation parameters
    static constexpr uint width = 512u;
    static constexpr uint height = 512u;
    static constexpr float dt = 1.0f;
    
    // Gray-Scott parameters (will be updated based on pattern type)
    float Du = 0.16f;   // Diffusion rate of U
    float Dv = 0.08f;   // Diffusion rate of V
    float F = 0.035f;   // Feed rate
    float k = 0.06f;    // Kill rate

    // Create ping-pong buffers for chemical concentrations
    Image<float> u_prev = device.create_image<float>(PixelStorage::FLOAT1, width, height);
    Image<float> v_prev = device.create_image<float>(PixelStorage::FLOAT1, width, height);
    Image<float> u_curr = device.create_image<float>(PixelStorage::FLOAT1, width, height);
    Image<float> v_curr = device.create_image<float>(PixelStorage::FLOAT1, width, height);

    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    // Initialize with random noise and a central seed
    std::mt19937 rng{std::random_device{}()};
    luisa::vector<float> u_init(width * height, 1.0f);  // U starts at 1.0 everywhere
    luisa::vector<float> v_init(width * height, 0.0f);  // V starts at 0.0

    // Add seed in the center
    int cx = width / 2;
    int cy = height / 2;
    int seed_radius = 20;
    for (int y = cy - seed_radius; y < cy + seed_radius; y++) {
        for (int x = cx - seed_radius; x < cx + seed_radius; x++) {
            if (x >= 0 && x < (int)width && y >= 0 && y < (int)height) {
                int dx = x - cx;
                int dy = y - cy;
                if (dx*dx + dy*dy < seed_radius*seed_radius) {
                    u_init[y * width + x] = 0.5f;
                    v_init[y * width + x] = 0.25f;
                }
            }
        }
    }

    // Add some random noise
    for (uint i = 0; i < width * height; i++) {
        u_init[i] += (rng() / float(UINT32_MAX)) * 0.05f;
        v_init[i] += (rng() / float(UINT32_MAX)) * 0.05f;
    }

    stream << u_prev.copy_from(u_init.data())
           << v_prev.copy_from(v_init.data())
           << synchronize();

    // Reaction-diffusion step kernel
    // Implements the Gray-Scott model using finite differences
    Kernel2D rd_step = [&](ImageFloat u_in, ImageFloat v_in, 
                           ImageFloat u_out, ImageFloat v_out,
                           Float feed_rate, Float kill_rate, 
                           Float diff_u, Float diff_v) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();

        // Helper to sample with toroidal wrapping
        auto sample_u = [&](Int2 offset) noexcept {
            Int2 p = make_int2(uv);
            Int2 s = make_int2(size);
            Int2 q = p + offset + s;
            Var sample_uv = make_uint2(q) % size;
            return u_in.read(sample_uv).x;
        };
        auto sample_v = [&](Int2 offset) noexcept {
            Int2 p = make_int2(uv);
            Int2 s = make_int2(size);
            Int2 q = p + offset + s;
            Var sample_uv = make_uint2(q) % size;
            return v_in.read(sample_uv).x;
        };

        // Read current values
        Var u = u_in.read(uv).x;
        Var v = v_in.read(uv).x;

        // Compute Laplacian using 5-point stencil
        // laplacian = (center + neighbors) / 5 - center
        Var u_lap = (sample_u(make_int2(-1, 0)) +
                     sample_u(make_int2(1, 0)) +
                     sample_u(make_int2(0, -1)) +
                     sample_u(make_int2(0, 1)) +
                     u) * 0.2f - u;
        
        Var v_lap = (sample_v(make_int2(-1, 0)) +
                     sample_v(make_int2(1, 0)) +
                     sample_v(make_int2(0, -1)) +
                     sample_v(make_int2(0, 1)) +
                     v) * 0.2f - v;

        // Gray-Scott reaction terms
        Var reaction = u * v * v;  // u*v^2
        
        // Update equations
        Var u_new = u + dt * (diff_u * u_lap - reaction + feed_rate * (1.0f - u));
        Var v_new = v + dt * (diff_v * v_lap + reaction - (feed_rate + kill_rate) * v);

        // Clamp to valid range
        u_new = clamp(u_new, 0.0f, 1.0f);
        v_new = clamp(v_new, 0.0f, 1.0f);

        u_out.write(uv, make_float4(u_new, 0.0f, 0.0f, 1.0f));
        v_out.write(uv, make_float4(v_new, 0.0f, 0.0f, 1.0f));
    };

    auto rd_shader = device.compile(rd_step);

    // Visualization kernel - maps chemical concentrations to colors
    Kernel2D visualize = [&](ImageFloat u_img, ImageFloat v_img, ImageFloat output) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();

        Var u = u_img.read(uv).x;
        Var v = v_img.read(uv).x;

        // Color mapping based on chemical concentrations
        // U (inhibitor) = blue channel
        // V (activator) = red/yellow channel
        
        // Create a gradient from blue (high U) to red/yellow (high V)
        Var blue = u * 0.8f;
        Var red = v * 1.5f;
        Var green = v * u * 0.5f;

        // Add some contrast
        red = pow(red, 0.7f);
        blue = pow(blue, 0.8f);

        Var color = make_float3(
            clamp(red, 0.0f, 1.0f),
            clamp(green, 0.0f, 1.0f),
            clamp(blue, 0.0f, 1.0f)
        );

        output.write(uv, make_float4(color, 1.0f));
    };

    auto visualize_shader = device.compile(visualize);

    // Clear kernel
    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        set_block_size(16, 16, 1);
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };
    auto clear = device.compile(clear_kernel);

    // Setup window and swapchain
    static constexpr uint display_scale = 2u;
    static constexpr uint display_width = width * display_scale;
    static constexpr uint display_height = height * display_scale;

    Window window{"Reaction-Diffusion (Gray-Scott)", make_uint2(display_width, display_height)};
    Swapchain swap_chain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = window.size(),
            .wants_vsync = true,
        });
    Image<float> display = device.create_image<float>(swap_chain.backend_storage(), window.size());
    Image<float> temp_display = device.create_image<float>(PixelStorage::FLOAT4, width, height);

    // Upscaling kernel (nearest neighbor)
    Kernel2D upscale_kernel = [](ImageFloat input, ImageFloat output) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var input_size = make_uint2(dispatch_size().x / 2u, dispatch_size().y / 2u);
        Var sample_uv = uv / 2u;
        output.write(uv, input.read(sample_uv));
    };
    auto upscale = device.compile(upscale_kernel);

    // Main simulation loop
    Clock clock;
    bool running = true;
    int pattern_type = 1;  // 1=Coral, 2=Fingerprint, 3=Spots, 4=Stripes

    while (!window.should_close()) {
        window.poll_events();

        // Handle pattern selection
        if (window.is_key_down(KEY_1)) {
            pattern_type = 1;  // Coral
            F = 0.035f;
            k = 0.06f;
        }
        if (window.is_key_down(KEY_2)) {
            pattern_type = 2;  // Fingerprint
            F = 0.037f;
            k = 0.06f;
        }
        if (window.is_key_down(KEY_3)) {
            pattern_type = 3;  // Spots
            F = 0.03f;
            k = 0.062f;
        }
        if (window.is_key_down(KEY_4)) {
            pattern_type = 4;  // Stripes
            F = 0.03f;
            k = 0.054f;
        }
        if (window.is_key_down(KEY_R)) {
            // Reset simulation
            stream << u_prev.copy_from(u_init.data())
                   << v_prev.copy_from(v_init.data())
                   << synchronize();
        }
        if (window.is_key_down(KEY_SPACE)) {
            running = !running;
        }

        // Update simulation
        if (running) {
            // Run multiple steps per frame for faster evolution
            for (int step = 0; step < 10; step++) {
                stream << rd_shader(u_prev, v_prev, u_curr, v_curr, F, k, Du, Dv)
                              .dispatch(width, height);
                std::swap(u_prev, u_curr);
                std::swap(v_prev, v_curr);
            }
        }

        // Render
        stream << visualize_shader(u_prev, v_prev, temp_display).dispatch(width, height)
               << upscale(temp_display, display).dispatch(display_width, display_height)
               << swap_chain.present(display);

    }

    stream << synchronize();
}
