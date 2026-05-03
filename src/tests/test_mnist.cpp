// MNIST digit recognition demo with interactive drawing.
// Demonstrates CUDA/Vulkan interop, GPU-GPU communication,
// and real-time machine learning inference integration.

#include <luisa/backends/ext/dx_hdr_ext.hpp>
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/gui/window.h>
#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/vk_cuda_interop.h>
#include <luisa/backends/ext/dx_cuda_interop.h>

using namespace luisa;
using namespace luisa::compute;

// Interop event type for synchronization between CUDA and graphics devices
using InteropEvent = luisa::variant<
    DxCudaTimelineEvent,
    TimelineEvent>;

// Function declarations for interop operations
InteropEvent create_interop_event(Device &cuda_device, Device &render_device);
Buffer<float> create_interop_buffer(Device &render_device, uint64_t size);
Buffer<float> get_cuda_buffer(Device &torch_device, Device &render_device, Buffer<float> &render_buffer, uint64_t &cuda_ptr, uint64_t &cuda_handle);
void unmap_interop_handle(Device &render_device, uint64_t cuda_ptr, uint64_t handle);

// CUDA device configuration extension for Vulkan interop
struct CUDADeviceConfigExtImpl : public CUDADeviceConfigExt {
    ExternalVkDevice external_device;
    [[nodiscard]] ExternalVkDevice get_external_vk_device() const noexcept override {
        return external_device;
    }
};

// Convert wide string to regular string
luisa::string from_wstr(const wchar_t *arg) {
    auto start = arg;
    while (*arg != 0) {
        ++arg;
    }
    auto size = (reinterpret_cast<size_t>(arg) - reinterpret_cast<size_t>(start)) / sizeof(wchar_t);
    luisa::string str;
    str.resize(size);
    for (size_t i = 0; i < size; ++i) {
        str[i] = (char)start[i];
    }
    return str;
}

// Main MNIST interpreter class handling rendering and ML interop
struct MnistInterpreter {
    Context context;
    Device torch_device;      // CUDA device for ML inference
    Device render_device;     // Graphics device for rendering
    Stream stream;            // Graphics stream
    Stream cuda_stream;       // CUDA stream
    Window window;
    Shader2D<Buffer<float>, float2, float2, bool> capsule_sdf_shader;
    Shader2D<Buffer<float>, Image<float>, float> hdr2ldr_shader;
    float2 last_mouse_pos{};
    float2 curr_mouse_pos{};
    InteropEvent interop_event;
    bool pressed = false;
    Swapchain swap_chain;
    Buffer<float> draw_buffer;
    Image<float> display_img;
    CommandList cmdlist;
    static constexpr uint2 display_resolution = make_uint2(1024u);
    static constexpr uint2 mnist_resolution = make_uint2(28u);
    
    MnistInterpreter(luisa::string_view context_dir, luisa::string_view backend_name)
        : context(context_dir),
          render_device(context.create_device(backend_name)),
          stream(render_device.create_stream(StreamTag::GRAPHICS)),
          window{"mnist", display_resolution}

    {
        // Configure CUDA device for interop
        DeviceConfig cuda_settings;
        {
            auto interop_ext = render_device.extension<VkCudaInterop>();
            if (interop_ext) {
                auto ext_device = luisa::make_unique<CUDADeviceConfigExtImpl>();
                ext_device->external_device = interop_ext->get_external_vk_device();
                cuda_settings = DeviceConfig{
                    .extension = std::move(ext_device),
                    .device_index = static_cast<size_t>(interop_ext->cuda_device_index())};
            }
        }

        torch_device = context.create_device("cuda", &cuda_settings);
        cuda_stream = torch_device.create_stream();
        interop_event = create_interop_event(torch_device, render_device);
        
        // Cubic Hermite interpolation callable
        Callable cubic_hermite{[](Float A, Float B, Float C, Float D, Float t) {
            Float t2 = t * t;
            Float t3 = t * t * t;
            Float a = -A / 2.0f + (3.0f * B) / 2.0f - (3.0f * C) / 2.0f + D / 2.0f;
            Float b = A - (5.0f * B) / 2.0f + 2.0f * C - D / 2.0f;
            Float c = -A / 2.0f + C / 2.0f;
            Float d = B;

            return a * t3 + b * t2 + c * t + d;
        }};

        // Point sampling callable for texture lookup
        Callable point_sample = [](BufferVar<float> img, UInt2 tex_size, Float2 uv) {
            auto coord = make_uint2(clamp(uv * make_float2(tex_size.x.cast<float>(), tex_size.y.cast<float>()), float2(0.f), make_float2(tex_size) - 0.5f));
            return saturate(img.read(coord.x + tex_size.x * coord.y));
        };
        
        // Bicubic Hermite texture sampling - https://www.shadertoy.com/view/MllSzX
        Callable bicubic_hermite_texture_sample([&](BufferVar<float> img, Float2 P) {
            auto c_textureSize = (float)mnist_resolution.x;
            auto c_onePixel = 1.f / c_textureSize;
            auto c_twoPixels = 2.f / c_textureSize;
            Float2 pixel = P * make_float2(c_textureSize) + 0.5f;

            Float2 frac = fract(pixel);
            pixel = floor(pixel) / make_float2(c_textureSize) - make_float2(c_onePixel / 2.0f);

            Float C00 = point_sample(img, mnist_resolution, pixel + make_float2(-c_onePixel, -c_onePixel));
            Float C10 = point_sample(img, mnist_resolution, pixel + make_float2(0.0f, -c_onePixel));
            Float C20 = point_sample(img, mnist_resolution, pixel + make_float2(c_onePixel, -c_onePixel));
            Float C30 = point_sample(img, mnist_resolution, pixel + make_float2(c_twoPixels, -c_onePixel));

            Float C01 = point_sample(img, mnist_resolution, pixel + make_float2(-c_onePixel, 0.0f));
            Float C11 = point_sample(img, mnist_resolution, pixel + make_float2(0.0f, 0.0f));
            Float C21 = point_sample(img, mnist_resolution, pixel + make_float2(c_onePixel, 0.0f));
            Float C31 = point_sample(img, mnist_resolution, pixel + make_float2(c_twoPixels, 0.0f));

            Float C02 = point_sample(img, mnist_resolution, pixel + make_float2(-c_onePixel, c_onePixel));
            Float C12 = point_sample(img, mnist_resolution, pixel + make_float2(0.0f, c_onePixel));
            Float C22 = point_sample(img, mnist_resolution, pixel + make_float2(c_onePixel, c_onePixel));
            Float C32 = point_sample(img, mnist_resolution, pixel + make_float2(c_twoPixels, c_onePixel));

            Float C03 = point_sample(img, mnist_resolution, pixel + make_float2(-c_onePixel, c_twoPixels));
            Float C13 = point_sample(img, mnist_resolution, pixel + make_float2(0.0f, c_twoPixels));
            Float C23 = point_sample(img, mnist_resolution, pixel + make_float2(c_onePixel, c_twoPixels));
            Float C33 = point_sample(img, mnist_resolution, pixel + make_float2(c_twoPixels, c_twoPixels));

            Float CP0X = cubic_hermite(C00, C10, C20, C30, frac.x);
            Float CP1X = cubic_hermite(C01, C11, C21, C31, frac.x);
            Float CP2X = cubic_hermite(C02, C12, C22, C32, frac.x);
            Float CP3X = cubic_hermite(C03, C13, C23, C33, frac.x);

            return cubic_hermite(CP0X, CP1X, CP2X, CP3X, frac.y);
        });

        // Signed distance function for capsule (drawing strokes)
        Callable sd_capsule{[&](Float2 p, Float2 a, Float2 b, Float r) {
            Float2 pa = p - a;
            Float2 ba = b - a;
            Float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
            return length(pa - ba * h) - r;
        }};
        
        // Linear to sRGB color space conversion
        Callable linear_to_srgb = [&](Var<float3> x) noexcept {
            return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                                   12.92f * x,
                                   x <= 0.00031308f));
        };
        
        // Kernel for drawing capsule SDF (drawing strokes)
        Kernel2D capsule_sdf = [&](BufferVar<float> hdr_img, Float2 last_pos, Float2 pos, Bool clear) {
            UInt2 coord = dispatch_id().xy();
            auto idx = coord.x + coord.y * mnist_resolution.x;
            $if (clear) {
                hdr_img.write(idx, 0.f);
            }
            $else {
                auto uv = (make_float2(coord) + 0.5f) / make_float2(dispatch_size().xy());
                auto dist = 1.f - saturate(sd_capsule(uv, last_pos, pos, 0.01f) * 0.8f * mnist_resolution.x);
                hdr_img.write(idx, max(hdr_img.read(idx), dist));
            };
        };
        
        // Kernel for HDR to LDR conversion
        Kernel2D hdr2ldr_kernel = [&](BufferVar<float> hdr_image, ImageFloat ldr_image, Float scale) noexcept {
            set_name("hdr2ldr_kernel");
            UInt2 coord = dispatch_id().xy();
            Float4 hdr = make_float4(make_float3(bicubic_hermite_texture_sample(hdr_image, (make_float2(coord) + 0.5f) / make_float2(dispatch_size().xy()))), 1.f);
            Float3 ldr = linear_to_srgb(clamp(hdr.xyz() / hdr.w * scale, 0.f, 1.f));
            ldr_image.write(coord, make_float4(ldr, 1.f));
        };
        
        capsule_sdf_shader = render_device.compile(capsule_sdf);
        hdr2ldr_shader = render_device.compile(hdr2ldr_kernel);
        
        // Mouse interaction callbacks
        window.set_mouse_callback([&](MouseButton button, Action action, float2 xy) {
            if (button != MouseButton::MOUSE_BUTTON_1 && button != MouseButton::MOUSE_BUTTON_2) return;
            if (action == Action::ACTION_PRESSED) {
                curr_mouse_pos = xy;
                last_mouse_pos = xy;
                pressed = true;
            } else if (action == Action::ACTION_RELEASED) {
                pressed = false;
            }
        });
        window.set_cursor_position_callback([&](float2 xy) {
            if (pressed) {
                last_mouse_pos = curr_mouse_pos;
                curr_mouse_pos = xy;
            }
        });
        
        // Create swapchain and resources
        swap_chain = render_device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window.native_display(),
                .window = window.native_handle(),
                .size = make_uint2(display_resolution),
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 1,
            });
        draw_buffer = create_interop_buffer(render_device, mnist_resolution.x * mnist_resolution.y);
        display_img = render_device.create_image<float>(swap_chain.backend_storage(), display_resolution);
    }
    
    uint64_t _evt_fence{1};
    bool clear_next_frame{true};
    bool enter_pressed{false};
    
    // Update frame, handle input, and manage GPU-GPU communication
    bool update(uint64_t cuda_data_buffer_ptr, uint64_t input_ptr) {
        window.poll_events();
        bool update = false;
        
        // Clear buffer if requested
        if (clear_next_frame) {
            clear_next_frame = false;
            cmdlist << capsule_sdf_shader(draw_buffer, float2(100), float2(100), true).dispatch(mnist_resolution);
        }
        
        // Handle keyboard input
        if (window.is_key_down(Key::KEY_SPACE)) {
            clear_next_frame = true;
        }
        if (window.is_key_down(Key::KEY_ENTER) || window.is_key_down(Key::KEY_KP_ENTER)) {
            if (!enter_pressed) {
                update = true;
            }
            enter_pressed = true;
        } else {
            enter_pressed = false;
        }
        
        // Draw stroke if mouse is pressed
        if (pressed) {
            cmdlist << capsule_sdf_shader(draw_buffer, last_mouse_pos / make_float2(display_resolution), curr_mouse_pos / make_float2(display_resolution), false).dispatch(mnist_resolution);
        }
        
        // Render to display
        cmdlist << hdr2ldr_shader(draw_buffer, display_img, 2.f).dispatch(display_resolution);
        stream << cmdlist.commit() << swap_chain.present(display_img);
        
        uint64_t cuda_ptr, cuda_handle;
        
        // Copy torch tensor to display buffer
        if (input_ptr != 0) {
            auto cuda_buffer = get_cuda_buffer(torch_device, render_device, draw_buffer, cuda_ptr, cuda_handle);
            auto torch_tensor_buffer = torch_device.import_external_buffer<float>((void *)input_ptr, draw_buffer.size());
            cmdlist << cuda_buffer.copy_from(torch_tensor_buffer);
            // Deferred deallocate external buffer and unmap memory after execution
            cmdlist.add_callback([this, cuda_buffer = std::move(cuda_buffer), cuda_ptr, cuda_handle]() {
                unmap_interop_handle(render_device, cuda_ptr, cuda_handle);
            });
            cuda_stream << cmdlist.commit();
            
            // Let render stream wait for CUDA inference stream
            luisa::visit(
                [&]<typename T>(T const &t) {
                    // Vulkan device
                    if constexpr (std::is_same_v<T, TimelineEvent>) {
                        cuda_stream << t.signal(_evt_fence);
                        stream << render_device.extension<VkCudaInterop>()->vk_wait(t, _evt_fence);
                    } else {
                        cuda_stream << t.cuda_signal(_evt_fence);
                        stream << t.dx_wait(_evt_fence);
                    }
                    ++_evt_fence;
                },
                interop_event);
            cuda_ptr = 0;
            cuda_handle = 0;
        }
        
        // Copy display buffer to torch tensor
        if (update) {
            // Let CUDA inference stream wait for render stream
            luisa::visit(
                [&]<typename T>(T const &t) {
                    // Vulkan device
                    if constexpr (std::is_same_v<T, TimelineEvent>) {
                        stream << render_device.extension<VkCudaInterop>()->vk_signal(t, _evt_fence);
                        cuda_stream << t.wait(_evt_fence);
                    } else {
                        stream << t.dx_signal(_evt_fence);
                        cuda_stream << t.cuda_wait(_evt_fence);
                    }
                    ++_evt_fence;
                },
                interop_event);
            
            // Import tensor's data pointer to buffer
            auto torch_tensor_buffer = torch_device.import_external_buffer<float>((void *)cuda_data_buffer_ptr, draw_buffer.size());
            // Make interop buffer's CUDA view
            auto cuda_buffer = get_cuda_buffer(torch_device, render_device, draw_buffer, cuda_ptr, cuda_handle);
            // Copy interop buffer to tensor
            cmdlist << torch_tensor_buffer.copy_from(cuda_buffer);
            // Unmap memory range after execution
            cmdlist.add_callback([this, cuda_buffer = std::move(cuda_buffer), cuda_ptr, cuda_handle]() {
                unmap_interop_handle(render_device, cuda_ptr, cuda_handle);
            });
            cuda_stream << cmdlist.commit();
        }
        return update;
    }
    
    ~MnistInterpreter() {
        stream.synchronize();
    }
};

// Global interpreter instance for C API
luisa::optional<MnistInterpreter> interpreter{};

// C API for external integration (e.g., Python/PyTorch)
LUISA_EXPORT_API void init(wchar_t const *runtime_dir, wchar_t const *backend_name) {
    interpreter.emplace(from_wstr(runtime_dir), from_wstr(backend_name));
}

LUISA_EXPORT_API bool should_close() {
    return !interpreter || interpreter->window.should_close();
}

LUISA_EXPORT_API bool update_frame(uint64_t cuda_ptr, uint64_t input_cuda_ptr) {
    if (!interpreter) return false;
    return interpreter->update(cuda_ptr, input_cuda_ptr);
}

LUISA_EXPORT_API void dispose() {
    interpreter.reset();
}

// Create interop buffer based on backend
Buffer<float> create_interop_buffer(Device &render_device, uint64_t size) {
    if (render_device.backend_name() == "vk") {
        auto vk_interop = render_device.extension<VkCudaInterop>();
        return vk_interop->create_buffer<float>(size);
    } else if (render_device.backend_name() == "dx") {
        auto dx_interop = render_device.extension<DxCudaInterop>();
        return dx_interop->create_buffer<float>(size);
    } else {
        LUISA_ERROR("Unsupported backend {}", render_device.backend_name());
        return {};
    }
}

// Get CUDA-accessible buffer view for interop
Buffer<float> get_cuda_buffer(Device &torch_device, Device &render_device, Buffer<float> &render_buffer, uint64_t &cuda_ptr, uint64_t &cuda_handle) {
    cuda_handle = 0;
    cuda_ptr = 0;
    if (render_device.backend_name() == "vk") {
        auto vk_interop = render_device.extension<VkCudaInterop>();
        vk_interop->cuda_buffer(render_buffer.handle(), &cuda_ptr, &cuda_handle);
        return torch_device.import_external_buffer<float>(reinterpret_cast<void *>(cuda_ptr), render_buffer.size());
    } else if (render_device.backend_name() == "dx") {
        auto dx_interop = render_device.extension<DxCudaInterop>();
        dx_interop->cuda_buffer(render_buffer.handle(), &cuda_ptr, &cuda_handle);
        return torch_device.import_external_buffer<float>(reinterpret_cast<void *>(cuda_ptr), render_buffer.size());
    } else {
        LUISA_ERROR("Unsupported backend {}", render_device.backend_name());
        return {};
    }
}

// Unmap interop memory handle
void unmap_interop_handle(Device &render_device, uint64_t cuda_ptr, uint64_t handle) {
    if (handle == 0) return;
    if (render_device.backend_name() == "vk") {
        auto vk_interop = render_device.extension<VkCudaInterop>();
        vk_interop->unmap(reinterpret_cast<void *>(cuda_ptr), reinterpret_cast<void *>(handle));
    } else if (render_device.backend_name() == "dx") {
        auto dx_interop = render_device.extension<DxCudaInterop>();
        dx_interop->unmap(reinterpret_cast<void *>(cuda_ptr), reinterpret_cast<void *>(handle));
    }
}

// Create synchronization event for interop
InteropEvent create_interop_event(Device &cuda_device, Device &render_device) {
    if (render_device.backend_name() == "vk") {
        auto vk_interop = render_device.extension<VkCudaInterop>();
        return cuda_device.create_timeline_event();
    } else if (render_device.backend_name() == "dx") {
        auto dx_interop = render_device.extension<DxCudaInterop>();
        return dx_interop->create_timeline_event();
    } else {
        LUISA_ERROR("Unsupported backend {}", render_device.backend_name());
        return {};
    }
}
