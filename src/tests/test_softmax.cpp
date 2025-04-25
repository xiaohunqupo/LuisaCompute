#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/shared.h>
#include <luisa/core/clock.h>
#include <luisa/vstl/meta_lib.h>
#include <luisa/vstl/common.h>

using namespace luisa;
using namespace luisa::compute;

struct DispatchPack {
    Kernel2D<void(Buffer<float>)> kernel;
    uint2 dispatch_size;
};
DispatchPack softmax_kernel(uint2 size) {
    if (size.x > 2048) {
        LUISA_ERROR("Softmax size can not be larger than 2048");
    }
    if (any(size == 0u)) {
        LUISA_ERROR("Softmax size can not be 0");
    }
    auto block_size = next_pow2(size.x);
    block_size = std::max<uint>(block_size / 2u, 32u);
    auto kernel = Kernel2D([=](BufferVar<float> input) {
        set_block_size(block_size, 1, 1);
        Shared<float> shared_arr(block_size);
        auto thd_id = thread_id().x;
        Float value;
        auto id = dispatch_id().x * 2;
        $if (id < size.x) {
            value = exp(Float(input.read(id + size.x * dispatch_id().y)));
        }
        $else {
            value = 0.0f;
        };
        id += 1;
        $if (id < size.x) {
            value += exp(Float(input.read(id + size.x * dispatch_id().y)));
        };
        id -= 1;
        shared_arr[thd_id] = value;
        UInt thd_size = block_size / 2u;
        sync_block();
        $while (thd_size > 0) {
            $if (thd_id < thd_size) {
                value = shared_arr[thd_id * 2] + shared_arr[thd_id * 2 + 1];
            };
            sync_block();
            $if (thd_id < thd_size) {
                shared_arr[thd_id] = value;
            };
            thd_size /= 2u;
            sync_block();
        };
        auto write_func = [&]() {
            $if (id < size.x) {
                auto write_id = id + size.x * dispatch_id().y;
                input.write(write_id, exp(Float(input.read(write_id))) / shared_arr[0]);
            };
        };
        write_func();
        id += 1;
        write_func();
    });
    return DispatchPack{
        .kernel = std::move(kernel),
        .dispatch_size = uint2((size.x + block_size - 1u) & (~(block_size - 1u)), size.y)};
}

int main(int argc, char *argv[]) {
    auto ctx = Context(argv[0]);
    auto device = ctx.create_device("dx");
    auto stream = device.create_stream();
    const auto size = 3;
    auto pack = softmax_kernel(uint2(size, 1));
    auto shader = device.compile(pack.kernel);
    auto buffer = device.create_buffer<float>(size);
    luisa::vector<float> f(size);
    f[0] = 2.0;
    f[1] = 1.0;
    f[2] = 0.1;
    stream << buffer.copy_from(f.data()) << shader(buffer).dispatch(pack.dispatch_size) << buffer.copy_to(f.data()) << synchronize();
    float sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += f[i];
        LUISA_INFO("{}", f[i]);
    }
    LUISA_INFO("sum {}", sum);
    return 0;
}