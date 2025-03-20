#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/clock.h>
#include <luisa/vstl/meta_lib.h>
#include <fstream>
using namespace luisa;
using namespace luisa::compute;

template<typename T>
Kernel1D<void(Buffer<T>, Buffer<T>, Buffer<T>)> conv_1d_kernel(uint filter_size, uint start_padding, uint end_padding, uint stride_size) {
    using VarType = typename decltype([&]() {
        if constexpr (std::is_same_v<T, float>) {
            return vstd::TypeOf<Float>{};
        } else {
            return vstd::TypeOf<Half>{};
        }
    }())::Type;
    LUISA_ASSERT(end_padding <= filter_size);
    LUISA_ASSERT(start_padding <= filter_size);
    uint start_idx = filter_size - start_padding;
    auto kernel = Kernel1D([=](BufferVar<T> input, BufferVar<T> weights, BufferVar<T> output) {
        set_block_size(128, 1, 1);
        // device
        UInt3 id = dispatch_id();
        id.x *= (stride_size + 1);
        if (start_idx > 0) {
            id.x += start_idx;
        }
        Float r = 0.f;
        auto weight_offset = (filter_size * 2u + 1u) * id.y;
        for (auto i : dynamic_range(-int(filter_size), int(filter_size + 1))) {
            auto input_node_idx = i + Int(id.x);
            $if (input_node_idx >= 0 & input_node_idx < dispatch_size().x - end_padding) {
                Float input_val = Float(input.read(input_node_idx));
                auto weight_node_idx = i + filter_size;
                Float weight_val = Float(weights.read(weight_offset + weight_node_idx));
                r += input_val * weight_val;
            };
        }
        // TODO: fused activation
        output.write(dispatch_size().y * dispatch_id().x + id.y, VarType(r));
    });
    return kernel;
}

// template<typename T>
// auto create_tensor(Device &device, size_t element_size) {
//     return device.create_buffer<T>(element_size);
// }

int main(int argc, char *argv[]) {
    auto ctx = Context(argv[0]);
    auto device = ctx.create_device("dx");
    auto stream = device.create_stream();
    auto shader = device.compile(conv_1d_kernel<half>(3, 3, 3, 0));
    // auto a_buffer = create_tensor<half>(device, batch_size * node_size * node_size);
    // auto b_buffer = create_tensor<half>(device, node_size);
    // auto r_buffer = create_tensor<half>(device, batch_size * node_size);

    // auto kernel_pack = get_matrix_multiple_kernel<half>(uint2(node_size, node_size), uint2(1, node_size), batch_size, true, false);
    // auto shader = device.compile(kernel_pack.kernel);
    // Clock clk;
    // LUISA_INFO("Dispatch size {}", kernel_pack.dispatch_size);
    // stream
    //     << shader(a_buffer, b_buffer, r_buffer).dispatch(kernel_pack.dispatch_size)
    //     << synchronize();
    // CommandList cmdlist;
    // for (int i = 0; i < 5; ++i) {
    //     cmdlist << shader(a_buffer, b_buffer, r_buffer).dispatch(kernel_pack.dispatch_size);
    // }
    // clk.tic();
    // stream << cmdlist.commit() << synchronize();
    // auto time = clk.toc();
    // LUISA_INFO("Final time {}", time);
}