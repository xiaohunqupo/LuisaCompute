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
            auto out_tensor = Tensor::gemm(t0, t1, FusedActivation::relu(), TensorElementType::Float32);
        });
    kernel.compile(
        TensorDescriptor{
            std::initializer_list<size_t>{4ull, 4ull, 1ull},
            TensorElementType::Float32},
        TensorDescriptor{std::initializer_list<size_t>{4ull, 4ull, 1ull}, TensorElementType::Float32});
    auto b0 = device.create_buffer<float>(16);
    auto b1 = device.create_buffer<float>(16);
    CommandList cmdlist;
    kernel.execute(cmdlist, b0, b1);
    stream << cmdlist.commit() << synchronize();
}