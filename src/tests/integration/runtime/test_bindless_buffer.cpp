#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/gui/window.h>
#include "../../reference_image.h"

#include <filesystem>

using namespace luisa;
using namespace luisa::compute;

// contributed by @swifly in issue #67
int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    auto opts = luisa::test::ImageTestOptions::parse(argc, argv);
    Device device = context.create_device(argv[1]);

    constexpr uint2 resolution = make_uint2(1280, 720);

    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Image<float> device_image1 = device.create_image<float>(PixelStorage::BYTE4, resolution);
    BindlessArray bdls = device.create_bindless_array(65535);
    Buffer<float4> buffer = device.create_buffer<float4>(4);
    std::vector<float4> a{4};
    a[0] = {1, 0, 0, 1};
    a[1] = {0, 1, 0, 1};
    a[2] = {0, 0, 1, 1};
    a[3] = {1, 1, 1, 1};
    stream << buffer.copy_from(a.data()) << synchronize();
    bdls.emplace_on_update(5, buffer);
    stream << bdls.update() << synchronize();

    Kernel2D kernel = [&](Float time) {
        Var coord = dispatch_id().xy();
        UInt i2 = ((coord.x + cast<uint>(time)) / 16 % 4);
        auto vertex_array = bdls->buffer<float4>(5);
        Float4 p = vertex_array.read(i2);
        device_image1->write(coord, make_float4(p));
    };
    auto s = device.compile(kernel);
    auto ref_dir = luisa::test::find_reference_dir(std::filesystem::path{argv[0]}.parent_path());
    if (!opts.offline) {
        Window window{"Display", resolution};

        Swapchain swapchain = device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window.native_display(),
                .window = window.native_handle(),
                .size = resolution,
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 2,
            });
        Clock clk;
        while (!window.should_close()) {
            stream << s(static_cast<float>(clk.toc() * .05f))
                          .dispatch(1280, 720)
                   << swapchain.present(device_image1);
            window.poll_events();
        }
        return 0;
    } else {
        luisa::vector<std::byte> pixels(device_image1.view().size_bytes());
        stream << s(0.0f).dispatch(resolution.x, resolution.y)
               << device_image1.copy_to(pixels.data())
               << synchronize();
        auto result = luisa::test::save_and_compare(
            reinterpret_cast<const uint8_t *>(pixels.data()), static_cast<int>(resolution.x), static_cast<int>(resolution.y), 4,
            "test_bindless_buffer", opts.output_dir, ref_dir, opts.update_reference);
        LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
        if (!result.passed) {
            LUISA_ERROR("Reference comparison failed for test_bindless_buffer: {}", result.message);
            return 1;
        }
        return 0;
    }
}
