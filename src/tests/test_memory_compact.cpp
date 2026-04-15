#include <vector>
#include <cmath>

#include <luisa/core/logging.h>
#include <luisa/core/clock.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/volume.h>
#include <luisa/dsl/syntax.h>

#include <luisa/backends/ext/dx_config_ext.h>
#include <luisa/backends/ext/vk_config_ext.h>

using namespace luisa;
using namespace luisa::compute;

class DXConfigExtImpl final : public DirectXDeviceConfigExt {
public:
    luisa::move_only_function<void()> defragment_func;

    void GetDefragmentFunction(luisa::move_only_function<void()> &&defragment_func) override {
        this->defragment_func = std::move(defragment_func);
    }
};
class VKConfigExtImpl final : public VulkanDeviceConfigExt {
public:
    luisa::move_only_function<void()> defragment_func;

    void get_defragment_function(luisa::move_only_function<void()> &&defragment_func) override {
        this->defragment_func = std::move(defragment_func);
    }
};

int main(int argc, char *argv[]) {
    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: dx, vk", argv[0]);
        exit(1);
    }

    luisa::string backend = argv[1];
    bool use_dx = false;

    if (backend == "dx") {
        use_dx = true;
    } else if (backend == "vk") {
        use_dx = false;
    } else {
        LUISA_WARNING("This test requires either LUISA_TEST_DX_BACKEND or LUISA_TEST_VK_BACKEND to be defined");
        exit(0);
    }

    DeviceConfig config;
    luisa::move_only_function<void()> *defragment_func;

    if (use_dx) {
        auto dx_config_ext = luisa::make_unique<DXConfigExtImpl>();
        defragment_func = &dx_config_ext->defragment_func;
        config.extension = std::move(dx_config_ext);
    } else {
        auto vk_config_ext = luisa::make_unique<VKConfigExtImpl>();
        defragment_func = &vk_config_ext->defragment_func;
        config.extension = std::move(vk_config_ext);
    }

    Device device = context.create_device(backend, &config);
    Stream stream = device.create_stream();

    // Allocate resources: image, buffer, and volume
    LUISA_INFO("Allocating resources...");

    // Create images
    Image<float> image1 = device.create_image<float>(PixelStorage::FLOAT4, 512u, 512u);
    Image<float> image2 = device.create_image<float>(PixelStorage::BYTE4, 1024u, 1024u);

    // Create buffer
    Buffer<float> buffer = device.create_buffer<float>(1024u * 1024u);

    // Create volume
    Volume<float> volume = device.create_volume<float>(PixelStorage::FLOAT4, 128u, 128u, 128u);

    // Initialize resources with some data
    auto init_image_kernel = device.compile<2>([&](ImageFloat img) noexcept {
        Float2 coord = make_float2(dispatch_id().xy());
        Float2 size = make_float2(dispatch_size().xy());
        Float2 uv = coord / size;
        img.write(dispatch_id().xy(), make_float4(uv, 0.5f, 1.0f));
    });

    auto init_buffer_kernel = device.compile<1>([&](BufferFloat buf) noexcept {
        buf.write(dispatch_id().x, cast<float>(dispatch_id().x) * 0.01f);
    });

    auto init_volume_kernel = device.compile<3>([&](VolumeFloat vol) noexcept {
        Float3 coord = make_float3(dispatch_id().xyz());
        Float3 size = make_float3(dispatch_size().xyz());
        Float3 uv = coord / size;
        vol.write(dispatch_id().xyz(), make_float4(uv, 1.0f));
    });

    stream << init_image_kernel(image1).dispatch(512u, 512u)
           << init_image_kernel(image2).dispatch(1024u, 1024u)
           << init_buffer_kernel(buffer).dispatch(1024u * 1024u)
           << init_volume_kernel(volume).dispatch(128u, 128u, 128u)
           << synchronize();

    LUISA_INFO("Resources allocated and initialized");

    // Call defragment function if available
    (*defragment_func)();

    // Verify resources still work after defragmentation
    // Read back some data to verify
    std::vector<float> buffer_data(1024 * 1024);
    stream << buffer.copy_to(luisa::span{buffer_data})
           << synchronize();

    // Check a few values
    bool success = true;
    for (size_t i = 0; i < 10; ++i) {
        float expected = i * 0.01f;
        if (std::abs(buffer_data[i] - expected) > 1e-5f) {
            LUISA_ERROR("Buffer data mismatch at index {}: expected {}, got {}", i, expected, buffer_data[i]);
            success = false;
        }
    }

    if (success) {
        LUISA_INFO("Memory compact test passed!");
    }

    return success ? 0 : 1;
}
