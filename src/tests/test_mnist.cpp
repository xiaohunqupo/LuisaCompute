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
luisa::variant<TimelineEvent, DxCudaTimelineEvent> create_interop_event(Device &torch_device, Device &render_device);
Buffer<float> create_interop_buffer(Device &torch_device, Device &render_device, uint64_t size);

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    static constexpr uint2 display_resolution = make_uint2(1024u);
    static constexpr uint2 mnist_resolution = make_uint2(28u);
    Callable cubic_hermite{[](Float A, Float B, Float C, Float D, Float t) {
        Float t2 = t * t;
        Float t3 = t * t * t;
        Float a = -A / 2.0f + (3.0f * B) / 2.0f - (3.0f * C) / 2.0f + D / 2.0f;
        Float b = A - (5.0f * B) / 2.0f + 2.0f * C - D / 2.0f;
        Float c = -A / 2.0f + C / 2.0f;
        Float d = B;

        return a * t3 + b * t2 + c * t + d;
    }};

    Callable point_sample = [](BufferVar<float> img, UInt2 tex_size, Float2 uv) {
        auto coord = make_uint2(uv * make_float2(tex_size.x.cast<float>(), tex_size.y.cast<float>()));
        return img.read(coord.x + tex_size.x * coord.y);
    };
    // https://www.shadertoy.com/view/MllSzX
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

    Callable sd_capsule{[&](Float2 p, Float2 a, Float2 b, Float r) {
        Float2 pa = p - a;
        Float2 ba = b - a;
        Float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
        return length(pa - ba * h) - r;
    }};
    Callable linear_to_srgb = [&](Var<float3> x) noexcept {
        return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                               12.92f * x,
                               x <= 0.00031308f));
    };
    Kernel2D capsule_sdf = [&](BufferVar<float> hdr_img, Float2 last_pos, Float2 pos, Bool clear) {
        UInt2 coord = dispatch_id().xy();
        auto uv = (make_float2(coord) + 0.5f) / make_float2(dispatch_size().xy());
        auto dist = saturate(sd_capsule(uv, last_pos, pos, 0.01f) * 0.5f * mnist_resolution.x);
        auto idx = coord.x + coord.y * mnist_resolution.x;
        $if (clear) {
            hdr_img.write(idx, dist);
        }
        $else {
            hdr_img.write(idx, min(hdr_img.read(idx), dist));
        };
    };
    Kernel2D hdr2ldr_kernel = [&](BufferVar<float> hdr_image, ImageFloat ldr_image, Float scale) noexcept {
        set_name("hdr2ldr_kernel");
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = make_float4(make_float3(bicubic_hermite_texture_sample(hdr_image, (make_float2(coord) + 0.5f) / make_float2(dispatch_size().xy()))), 1.f);
        Float3 ldr = linear_to_srgb(clamp(hdr.xyz() / hdr.w * scale, 0.f, 1.f));
        ldr_image.write(coord, make_float4(ldr, 1.f));
    };
    auto capsule_sdf_shader = device.compile(capsule_sdf);
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);
    Window window{"mnist", display_resolution};
    float2 last_mouse_pos{};
    float2 curr_mouse_pos{};
    bool pressed = false;
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

    Swapchain swap_chain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = make_uint2(display_resolution),
            .wants_hdr = false,
            .wants_vsync = false,
            .back_buffer_count = 1,
        });
    auto draw_buffer = create_interop_buffer(device, device, mnist_resolution.x * mnist_resolution.y);
    Image<float> display_img = device.create_image<float>(swap_chain.backend_storage(), display_resolution);
    CommandList cmdlist;
    auto clear = [&] {
        cmdlist << capsule_sdf_shader(draw_buffer, float2(100), float2(100), true).dispatch(mnist_resolution);
    };
    clear();
    while (!window.should_close()) {
        window.poll_events();
        if (window.is_key_down(Key::KEY_SPACE)) {
            clear();
        }
        if (pressed) {
            cmdlist << capsule_sdf_shader(draw_buffer, last_mouse_pos / make_float2(display_resolution), curr_mouse_pos / make_float2(display_resolution), false).dispatch(mnist_resolution);
        }
        cmdlist << hdr2ldr_shader(draw_buffer, display_img, 2.f).dispatch(display_resolution);
        stream << cmdlist.commit() << swap_chain.present(display_img);
    }
    stream.synchronize();
}

luisa::variant<TimelineEvent, DxCudaTimelineEvent> create_interop_event(Device &torch_device, Device &render_device) {
    if (render_device.backend_name() == "cuda") {
        return render_device.create_timeline_event();
    } else if (render_device.backend_name() == "vk") {
        // Cuda is using vulkan's event api, so we can directly use cuda's event for vulkan
        return torch_device.create_timeline_event();
    } else if (render_device.backend_name() == "dx") {
        auto dx_interop = render_device.extension<DxCudaInterop>();
        return dx_interop->create_timeline_event();
    } else {
        LUISA_ERROR("Unsupported backend {}", render_device.backend_name());
        return {};
    }
}
Buffer<float> create_interop_buffer(Device &torch_device, Device &render_device, uint64_t size) {
    if (render_device.backend_name() == "cuda") {
        return render_device.create_buffer<float>(size);
    } else if (render_device.backend_name() == "vk") {
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
