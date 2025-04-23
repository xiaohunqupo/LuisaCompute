#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/clock.h>
#include <luisa/tensor/kernel.h>
#include <luisa/tensor/pass/expr_topo.h>
#include <luisa/tensor/tensor_interface.h>
using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    TensorBuilder builder;
    auto ctx = Context(argv[0]);
    auto device = ctx.create_device("dx");
    auto stream = device.create_stream();
    auto interface = TensorInterface::create_fallback_backend(device);
    auto kernel = load_tensor(
        interface.get(),
        [&](Tensor t0,
            Tensor t1) {
            // TODO: how can we get out_tensor
            return Tensor::matmul(t0, t1, FusedActivation::relu());
        });
    kernel.compile(
        TensorDescriptor{
            std::initializer_list<size_t>{4ull, 4ull, 1ull},
            TensorElementType::Float32},
        TensorDescriptor{std::initializer_list<size_t>{4ull, 4ull, 1ull}, TensorElementType::Float32});
    auto b0 = device.create_buffer<float>(16);
    auto b1 = device.create_buffer<float>(16);
    CommandList cmdlist;
    float4x4 sb = make_float4x4(
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16);
    float4x4 sb1 = make_float4x4(
        10, 20, 30, 40,
        50, 60, 70, 80,
        1, 2, 3, 4,
        1, 2, 3, 4);
    float4x4 result;
    cmdlist << b0.copy_from(&sb) << b1.copy_from(&sb1);
    auto outs = kernel.execute(cmdlist, b0, b1);
    auto bout = outs[0];
    cmdlist << bout.copy_to(&result);
    stream << cmdlist.commit() << synchronize();
    LUISA_INFO("{}", result);
}