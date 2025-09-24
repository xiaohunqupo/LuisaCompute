#include "transient_resource_device/transient_resource_device.h"
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/gui/window.h>
using namespace luisa;
using namespace luisa::compute;
int main(int argc, char *argv[]) {
    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    auto write_shader = device.compile<2>([](ImageVar<float> img, UInt2 offset, Float z_value) {
        auto uv = (make_float2(dispatch_id().xy()) + 0.5f) / make_float2(dispatch_size().xy());
        img.write(dispatch_id().xy() + offset, make_float4(uv, z_value, 1.0f));
    });
    static constexpr uint2 resolution = make_uint2(1024u);
    Window window{"path tracing", resolution};
    Swapchain swap_chain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = make_uint2(resolution),
            .wants_hdr = false,
            .wants_vsync = false,
            .back_buffer_count = 8,
        });
    luisa::compute::Device transient_res_device{luisa::make_unique<utils::TransientResourceDevice>(Context{context}, device.impl())};
    auto storage = swap_chain.backend_storage();
    auto dst_tex = device.create_image<float>(storage, resolution);
    {
        utils::TransientResourceDeviceScope managed_scope{
            stream,
            transient_res_device};
        // pass 1
        {
            auto tex = managed_scope.create_transient_image<float>("MyTexture", storage, resolution);
            managed_scope.cmdlist << write_shader(tex, uint2(0), 0.0f).dispatch(resolution.x, resolution.y / 2);
        }
        // pass 2
        {
            // exactly same texture as pass 1
            auto tex = managed_scope.create_transient_image<float>("MyTexture", storage, resolution);
            managed_scope.cmdlist << write_shader(tex, uint2(0, resolution.y / 2), 1.0f).dispatch(resolution.x, resolution.y / 2);
        }
        // pass 3
        {
            // exactly same texture as pass 1 and pass 2
            auto tex = managed_scope.create_transient_image<float>("MyTexture", storage, resolution);
            // mix-in usage of transient and REAL resources
            managed_scope.cmdlist << dst_tex.copy_from(tex);
        }
        // dispatch to stream after scope
    }
    while (!window.should_close()) {
        window.poll_events();
        stream << swap_chain.present(dst_tex);
    }
    stream.synchronize();
    return 0;
}