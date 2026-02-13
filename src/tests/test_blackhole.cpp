// Black Hole Renderer - Interstellar Style
// Renders a Schwarzschild black hole with gravitational lensing and an accretion disk.
// Features relativistic effects including light bending, Doppler beaming, and gravitational redshift.
//
// Features demonstrated:
// - Gravitational lensing via ray tracing with curved light paths
// - Accretion disk with temperature-based emission
// - Doppler shifting (redshift/blueshift) from orbital motion
// - Interactive camera controls

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

// Black hole parameters
static constexpr float bh_mass = 1.0f;           // Solar masses (scaled)
static constexpr float bh_radius = 2.0f;         // Schwarzschild radius (Rs = 2GM/c²)
static constexpr float photon_sphere = 3.0f;     // 1.5 * Rs
static constexpr float accretion_inner = 6.0f;   // Inner edge of accretion disk
static constexpr float accretion_outer = 15.0f;  // Outer edge of accretion disk

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    LUISA_INFO("Black Hole Renderer - Interstellar Style");
    LUISA_INFO("Controls: Mouse drag = Rotate, Scroll/+/- = Zoom, ESC = Quit");

    // Image dimensions
    static constexpr uint width = 1024u;
    static constexpr uint height = 1024u;

    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    // Setup window
    Window window{"Black Hole - Interstellar Style", make_uint2(width, height)};
    
    // Camera state
    float rot_x = 0.2f;  // Tilt angle
    float rot_y = 0.0f;  // Rotation around black hole
    float zoom = 30.0f;  // Camera distance
    bool mouse_down = false;
    float2 last_mouse_pos{0.0f, 0.0f};
    
    window.set_mouse_callback([&mouse_down, &last_mouse_pos](MouseButton, Action a, float2 p) noexcept {
        if (a == Action::ACTION_PRESSED) {
            mouse_down = true;
            last_mouse_pos = p;
        } else if (a == Action::ACTION_RELEASED) {
            mouse_down = false;
        }
    });
    
    window.set_cursor_position_callback([&mouse_down, &last_mouse_pos, &rot_x, &rot_y](float2 p) noexcept {
        if (mouse_down) {
            float dx = p.x - last_mouse_pos.x;
            float dy = p.y - last_mouse_pos.y;
            rot_y += dx * 0.005f;
            rot_x += dy * 0.005f;
            rot_x = clamp(rot_x, -1.0f, 1.0f);
            last_mouse_pos = p;
        }
    });
    
    window.set_scroll_callback([&zoom](float2 offset) noexcept {
        zoom *= (1.0f - offset.y * 0.1f);
        zoom = clamp(zoom, 10.0f, 100.0f);
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

    // Black hole rendering kernel with anti-aliasing
    Kernel2D blackhole_kernel = [&](ImageFloat image, Float rot_x, Float rot_y, Float cam_distance, Float time) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();
        
        // Anti-aliasing: 4x supersampling with jitter
        Var final_color = make_float3(0.0f);
        
        for (int sample_i = 0; sample_i < 4; sample_i++) {
            // Jitter offsets for 4x supersampling
            float2 offsets[4] = {{0.25f, 0.25f}, {0.75f, 0.25f}, {0.25f, 0.75f}, {0.75f, 0.75f}};
            Var offset = make_float2(offsets[sample_i][0], offsets[sample_i][1]);
            
            // Add sub-pixel jitter for smoother results
            Var jittered_uv = make_float2(uv) + offset;
            
            // Normalized coordinates (-1 to 1)
            Var ndc = (jittered_uv / make_float2(size)) * 2.0f - 1.0f;
            ndc.y = -ndc.y;  // Flip Y
            Var aspect = cast<float>(size.x) / cast<float>(size.y);
            ndc.x *= aspect;
            
            // Camera setup
            Var fov = 0.6f;
            
            // Ray origin (camera position)
            Var cam_pos = make_float3(
                cam_distance * sin(rot_y) * cos(rot_x),
                cam_distance * sin(rot_x),
                cam_distance * cos(rot_y) * cos(rot_x)
            );
            
            // Look at origin
            Var forward = normalize(-cam_pos);
            Var right = normalize(cross(forward, make_float3(0.0f, 1.0f, 0.0f)));
            Var up = cross(right, forward);
            
            // Ray direction through pixel
            Var ray_dir = normalize(forward + ndc.x * right * fov + ndc.y * up * fov);
            
            // Ray marching with gravitational bending
            Var pos = cam_pos;
            Var dir = ray_dir;
            Var sample_color = make_float3(0.0f);
            Var hit_disk = def(false);
            
            // Background starfield (procedural) - improved with better distribution
            auto get_star_color = [&](Float3 rd) noexcept {
                // Multiple octaves of noise for better star distribution
                Var star1 = fract(sin(rd.x * 123.456f + rd.y * 234.567f + rd.z * 345.678f) * 43758.5453f);
                Var star2 = fract(sin(rd.x * 456.789f + rd.y * 567.890f + rd.z * 678.901f) * 23421.1234f);
                Var star3 = fract(sin(rd.x * 789.012f + rd.y * 890.123f + rd.z * 901.234f) * 54321.6789f);
                
                // Different brightness thresholds for variety
                Var brightness = ite(star1 > 0.997f, 0.9f, 0.0f);
                brightness = ite((star2 > 0.998f) & (brightness < 0.1f), 0.7f, brightness);
                brightness = ite((star3 > 0.999f) & (brightness < 0.1f), 0.5f, brightness);
                
                // Subtle twinkle
                Var twinkle = 0.9f + 0.1f * sin(rd.x * 5.0f + time * 0.5f + rd.y * 3.0f);
                
                // Star color variation (blue-white to yellow-white)
                Var star_color = make_float3(
                    1.0f,
                    0.95f + 0.05f * star1,
                    0.8f + 0.2f * star2
                );
                
                return star_color * brightness * twinkle;
            };
        
        // Gravitational ray marching
        static constexpr int max_steps = 200;
        static constexpr float dt = 0.3f;
        
        $for (step, max_steps) {
            // Distance from black hole center
            Var r = length(pos);
            
            // Check if we hit the event horizon
            $if (r < bh_radius) {
                $break;  // Absorbed by black hole
            };
            
            // Check if we hit the accretion disk (thin disk in XZ plane)
            $if (abs(pos.y) < 0.3f & r > accretion_inner & r < accretion_outer) {
                // Check if we haven't hit disk yet
                $if (!hit_disk) {
                    hit_disk = true;
                    
                    // Orbital velocity at this radius (Keplerian)
                    Var orbital_speed = sqrt(bh_mass / r);
                    
                    // Disk temperature profile (T ~ r^(-3/4))
                    Var temp = 2.0f * pow(accretion_outer / r, 0.75f);
                    
                    // Doppler effect from orbital motion
                    // Disk orbits counter-clockwise when viewed from +Y
                    Var orbital_angle = atan2(pos.z, pos.x);
                    Var orbital_dir = make_float3(-sin(orbital_angle), 0.0f, cos(orbital_angle));
                    
                    // Doppler shift: approaching = blueshift (brighter), receding = redshift (dimmer)
                    Var doppler = dot(orbital_dir, dir);
                    Var beaming = pow(1.0f + doppler * orbital_speed * 2.0f, 2.0f);
                    
                    // Temperature shifted by Doppler
                    Var shifted_temp = temp * (1.0f + doppler * orbital_speed);
                    
                    // Blackbody color approximation
                    // Hot = white/blue, Cool = orange/red
                    Var disk_r = ite(shifted_temp > 1.0f, 1.0f, shifted_temp * 0.8f + 0.2f);
                    Var disk_g = ite(shifted_temp > 1.5f, 1.0f, shifted_temp * 0.6f);
                    Var disk_b = ite(shifted_temp > 2.0f, 1.0f, shifted_temp * 0.4f);
                    
                    Var disk_color = make_float3(disk_r, disk_g, disk_b);
                    
                    // Gravitational redshift (avoid negative inside sqrt)
                    Var redshift = sqrt(max(0.001f, 1.0f - bh_radius / max(r, bh_radius * 1.01f)));
                    disk_color *= redshift;
                    
                    // Apply intensity with Doppler beaming
                    Var intensity = beaming * smoothstep(accretion_outer, accretion_inner, r) * 
                                   smoothstep(accretion_inner * 0.9f, accretion_inner, r);
                    
                    sample_color += disk_color * intensity * 2.0f;
                };
            };
            
            // Gravitational acceleration (simplified)
            // For photons, the bending is twice Newtonian prediction (general relativity)
            Var r_safe = max(r, bh_radius * 0.5f);  // Prevent division by zero
            Var accel = -1.5f * bh_mass / (r_safe * r_safe * r_safe) * pos;  // 2x Newtonian for light
            
            // Update direction (bending)
            Var new_dir = dir + accel * dt;
            Var new_dir_len = length(new_dir);
            // Prevent NaN from zero-length direction
            dir = ite(new_dir_len > 0.001f, new_dir / new_dir_len, dir);
            
            // Step forward
            pos = pos + dir * dt;
        };
        
        // If we didn't hit anything, show starfield background
        $if (!hit_disk) {
            sample_color = get_star_color(dir);
        };
        
        // Add lensing glow around black hole
        // Calculate distance from black hole in screen space for glow effect
        Var to_bh = normalize(-cam_pos);
        Var cos_angle = dot(dir, to_bh);
        Var glow = exp(-(1.0f - cos_angle) * 50.0f) * 0.5f;
        
        // Check if ray passes near photon sphere
        Var impact_param = length(cross(pos, dir)) / max(length(pos), 0.001f);
        Var near_photon_sphere = smoothstep(photon_sphere * 1.2f, photon_sphere, impact_param);
        
        sample_color += make_float3(1.0f, 0.9f, 0.7f) * glow * near_photon_sphere * 0.3f;
        
        // Accumulate samples
        final_color += sample_color;
        }
        
        // Average the 4 samples
        final_color *= 0.25f;
        
        // Tone mapping to prevent clipping
        final_color = final_color / (final_color + 1.0f) * 1.5f;
        
        // Ensure no NaN values
        final_color = clamp(final_color, 0.0f, 1.0f);
        
        image.write(uv, make_float4(final_color, 1.0f));
    };

    auto blackhole_shader = device.compile(blackhole_kernel);

    // Main loop
    Clock app_clock;
    while (!window.should_close()) {
        window.poll_events();

        if (window.is_key_down(KEY_ESCAPE)) {
            break;
        }
        
        // Keyboard zoom
        if (window.is_key_down(KEY_EQUAL) || window.is_key_down(KEY_KP_ADD)) {
            zoom *= 0.98f;
            zoom = max(zoom, 10.0f);
        }
        if (window.is_key_down(KEY_MINUS) || window.is_key_down(KEY_KP_SUBTRACT)) {
            zoom *= 1.02f;
            zoom = min(zoom, 100.0f);
        }

        // Render
        float time = static_cast<float>(app_clock.toc() * 1e-3);
        stream << blackhole_shader(display, rot_x, rot_y, zoom, time).dispatch(width, height)
               << swap_chain.present(display);
    }

    stream << synchronize();
}
