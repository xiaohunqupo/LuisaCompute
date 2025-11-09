#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context ctx(argv[0]);
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = ctx.create_device(argv[1]);
    auto stream = device.create_stream();

    auto shader = device.compile<1>([]() noexcept {
        set_block_size(32u);
        auto result = warp_prefix_sum(1);
        device_log("{} -> {}", thread_x(), result);
    });

    stream << shader().dispatch(32u) << synchronize();
}
