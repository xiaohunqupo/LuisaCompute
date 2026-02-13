// 2D Wave Equation Simulation - Interactive Water Ripples
// Simulates realistic water ripples using the finite difference method.
// Click and drag to create waves, watch them propagate and interfere.
//
// Features demonstrated:
// - Finite difference method for PDEs
// - Three-buffer time integration scheme
// - Interactive mouse input via callbacks
// - Real-time water surface rendering with caustics

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

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    LUISA_INFO("2D Wave Equation Simulation - Interactive Water Ripples");
    LUISA_INFO("Controls: Click+Drag = Create waves, Space = Reset, ESC = Quit");

    // Simulation parameters - tuned for nice wave propagation
    static constexpr uint width = 512u;
    static constexpr uint height = 512u;
    // c^2 * dt^2 should be < 0.5 for stability (CFL condition)
    static constexpr float c = 0.2f;        // Wave speed (lower = slower waves)
    static constexpr float dt = 1.0f;       // Time step
    static constexpr float damping = 0.995f;// Slight damping for energy loss

    // Create three buffers for time integration
    Image<float> height_prev = device.create_image<float>(PixelStorage::FLOAT1, width, height);
    Image<float> height_curr = device.create_image<float>(PixelStorage::FLOAT1, width, height);
    Image<float> height_next = device.create_image<float>(PixelStorage::FLOAT1, width, height);

    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    // Clear kernel
    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        set_block_size(16, 16, 1);
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };
    auto clear = device.compile(clear_kernel);

    // Initialize
    stream << clear(height_prev).dispatch(width, height)
           << clear(height_curr).dispatch(width, height)
           << clear(height_next).dispatch(width, height)
           << synchronize();

    // Wave equation solver kernel
    Kernel2D wave_step = [&](ImageFloat prev, ImageFloat curr, ImageFloat next) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();

        Var u_curr = curr.read(uv).x;

        // Sample neighbors with clamping
        auto sample = [&](Int2 offset) noexcept {
            Int2 p = make_int2(uv) + offset;
            p.x = clamp(p.x, 0, cast<int>(size.x) - 1);
            p.y = clamp(p.y, 0, cast<int>(size.y) - 1);
            return curr.read(make_uint2(p)).x;
        };

        Var u_left = sample(make_int2(-1, 0));
        Var u_right = sample(make_int2(1, 0));
        Var u_up = sample(make_int2(0, -1));
        Var u_down = sample(make_int2(0, 1));

        Var laplacian = u_left + u_right + u_up + u_down - 4.0f * u_curr;
        Var u_prev = prev.read(uv).x;
        Var u_next = 2.0f * u_curr - u_prev + (c * c * dt * dt) * laplacian;
        u_next *= damping;
        u_next = clamp(u_next, -2.0f, 2.0f);

        next.write(uv, make_float4(u_next, 0.0f, 0.0f, 1.0f));
    };

    auto wave_shader = device.compile(wave_step);

    // Droplet drop kernel
    Kernel2D drop_droplet = [](ImageFloat height, UInt2 center, Float strength) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var dx = cast<float>(cast<int>(uv.x) - cast<int>(center.x));
        Var dy = cast<float>(cast<int>(uv.y) - cast<int>(center.y));
        Var dist_sq = dx * dx + dy * dy;
        Var radius = 15.0f;
        Var gaussian = exp(-dist_sq / (radius * radius * 0.5f));
        Var current = height.read(uv).x;
        height.write(uv, make_float4(current + strength * gaussian, 0.0f, 0.0f, 1.0f));
    };

    auto drop_shader = device.compile(drop_droplet);

    // Rendering kernel with caustics effect
    Kernel2D render_kernel = [](ImageFloat height, ImageFloat output, Float time) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();

        Var h = height.read(uv).x;

        auto sample = [&](Int2 offset) noexcept {
            Int2 p = make_int2(uv) + offset;
            p.x = clamp(p.x, 0, cast<int>(size.x) - 1);
            p.y = clamp(p.y, 0, cast<int>(size.y) - 1);
            return height.read(make_uint2(p)).x;
        };

        Var h_left = sample(make_int2(-2, 0));
        Var h_right = sample(make_int2(2, 0));
        Var h_up = sample(make_int2(0, -2));
        Var h_down = sample(make_int2(0, 2));

        Var slope_x = abs(h_right - h_left);
        Var slope_y = abs(h_down - h_up);
        Var slope = sqrt(slope_x * slope_x + slope_y * slope_y);

        Var deep_color = make_float3(0.0f, 0.1f, 0.3f);
        Var shallow_color = make_float3(0.0f, 0.4f, 0.6f);
        Var foam_color = make_float3(0.9f, 0.95f, 1.0f);
        Var highlight_color = make_float3(0.6f, 0.8f, 1.0f);

        Var t = clamp(h * 0.5f + 0.5f, 0.0f, 1.0f);
        Var base_color = lerp(deep_color, shallow_color, t);

        Var caustics = smoothstep(0.1f, 0.5f, slope);
        base_color = lerp(base_color, highlight_color, caustics * 0.5f);

        Var foam = smoothstep(0.5f, 1.0f, h);
        base_color = lerp(base_color, foam_color, foam * 0.7f);

        Var shimmer = sin(cast<float>(uv.x) * 0.1f + time * 2.0f) * sin(cast<float>(uv.y) * 0.1f + time * 1.5f);
        shimmer = (shimmer + 1.0f) * 0.5f;
        base_color = base_color * (0.9f + 0.1f * shimmer);

        output.write(uv, make_float4(base_color, 1.0f));
    };

    auto render = device.compile(render_kernel);

    // Setup window
    static constexpr uint display_scale = 2u;
    static constexpr uint display_width = width * display_scale;
    static constexpr uint display_height = height * display_scale;

    Window window{"Water Ripples - Click and Drag!", make_uint2(display_width, display_height)};

    // Mouse state for interaction
    bool mouse_down = false;
    float2 mouse_pos{0.0f, 0.0f};
    float2 last_mouse_pos{-1.0f, -1.0f};

    // Setup mouse callbacks
    window.set_mouse_callback([&mouse_down, &mouse_pos, &last_mouse_pos](MouseButton, Action a, float2 p) noexcept {
        if (a == Action::ACTION_PRESSED || a == Action::ACTION_REPEATED) {
            mouse_down = true;
            last_mouse_pos = make_float2(-1.0f, -1.0f);  // Reset for new stroke
        } else {
            mouse_down = false;
        }
        mouse_pos = p;
    });

    window.set_cursor_position_callback([&mouse_down, &mouse_pos](float2 p) noexcept {
        mouse_pos = p;
    });

    Swapchain swap_chain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = window.size(),
            .wants_vsync = true,
        });
    Image<float> display = device.create_image<float>(swap_chain.backend_storage(), window.size());

    // Upscaling kernel
    Kernel2D upscale_kernel = [](ImageFloat input, ImageFloat output) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var sample_uv = uv / 2u;
        output.write(uv, input.read(sample_uv));
    };
    auto upscale = device.compile(upscale_kernel);

    // Main simulation loop
    Clock clock;
    std::mt19937 rng{std::random_device{}()};
    float time = 0.0f;

    while (!window.should_close()) {
        window.poll_events();

        if (window.is_key_down(KEY_ESCAPE)) {
            break;
        }
        if (window.is_key_down(KEY_SPACE)) {
            stream << clear(height_prev).dispatch(width, height)
                   << clear(height_curr).dispatch(width, height)
                   << clear(height_next).dispatch(width, height);
        }

        // Handle mouse interaction - drop droplets when mouse is down
        if (mouse_down) {
            // Convert to simulation coordinates
            uint sim_x = clamp(cast<uint>(mouse_pos.x / display_scale), 0u, width - 1);
            uint sim_y = clamp(cast<uint>(mouse_pos.y / display_scale), 0u, height - 1);
            
            float drop_strength = -0.5f;
            
            // If we have a previous position, interpolate
            if (last_mouse_pos.x >= 0) {
                uint last_x = clamp(cast<uint>(last_mouse_pos.x / display_scale), 0u, width - 1);
                uint last_y = clamp(cast<uint>(last_mouse_pos.y / display_scale), 0u, height - 1);
                
                // Simple line interpolation
                int dx = (int)sim_x - (int)last_x;
                int dy = (int)sim_y - (int)last_y;
                int steps = max(abs(dx), abs(dy));
                
                for (int i = 0; i <= steps && i < 10; i++) {
                    float t = (steps == 0) ? 0.0f : (float)i / steps;
                    uint ix = last_x + cast<uint>(dx * t);
                    uint iy = last_y + cast<uint>(dy * t);
                    stream << drop_shader(height_curr, make_uint2(ix, iy), drop_strength * 0.7f)
                                  .dispatch(width, height);
                }
            } else {
                // Single drop
                stream << drop_shader(height_curr, make_uint2(sim_x, sim_y), drop_strength)
                              .dispatch(width, height);
            }
            
            last_mouse_pos = mouse_pos;
        }

        // Auto-drop occasional droplets
        if (rng() % 120u == 0u) {
            uint rain_x = 50 + rng() % (width - 100);
            uint rain_y = 50 + rng() % (height - 100);
            stream << drop_shader(height_curr, make_uint2(rain_x, rain_y), -0.3f)
                          .dispatch(width, height);
        }

        // Update wave simulation (multiple steps per frame)
        for (int step = 0; step < 4; step++) {
            stream << wave_shader(height_prev, height_curr, height_next).dispatch(width, height);
            std::swap(height_prev, height_curr);
            std::swap(height_curr, height_next);
        }

        // Render
        time += 0.016f;
        Image<float> temp_display = device.create_image<float>(PixelStorage::FLOAT4, width, height);
        stream << render(height_curr, temp_display, time).dispatch(width, height)
               << upscale(temp_display, display).dispatch(display_width, display_height)
               << swap_chain.present(display);

    }

    stream << synchronize();
}
