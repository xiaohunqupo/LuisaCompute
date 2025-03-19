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
struct DispatchPack {
    Kernel3D<void(Buffer<T>, Buffer<T>, Buffer<T>)> kernel;
    uint3 dispatch_size;
};

template<typename T>
DispatchPack<T> get_matrix_multiple_kernel(uint2 lhs_matrix_size, uint2 rhs_matrix_size, uint batch_size, bool lhs_batch, bool rhs_batch) {
    using VarType = typename decltype([&]() {
        if constexpr (std::is_same_v<T, float>) {
            return vstd::TypeOf<Float>{};
        } else {
            return vstd::TypeOf<Half>{};
        }
    }())::Type;
    auto iterate_size = std::min(lhs_matrix_size.x, rhs_matrix_size.y);
    auto bias_size = std::max(lhs_matrix_size.x, rhs_matrix_size.y);
    uint3 block_size;
    if (batch_size == 1) {
        block_size = make_uint3(8, 8, 1);
    } else if (batch_size < 4) {
        block_size = make_uint3(8, 4, batch_size);
    } else if (batch_size < 8) {
        block_size = make_uint3(4, 4, batch_size);
    } else if (batch_size < 16) {
        block_size = make_uint3(4, 2, batch_size);
    } else if (batch_size < 32) {
        block_size = make_uint3(2, 2, batch_size);
    } else if (batch_size < 64) {
        block_size = make_uint3(2, 1, batch_size);
    } else if (batch_size < 64ull * 65535ull) {
        block_size = make_uint3(1, 1, 64);
    } else if (batch_size < 128ull * 65535ull) {
        block_size = make_uint3(1, 1, 128);
    } else if (batch_size < 256ull * 65535ull) {
        block_size = make_uint3(1, 1, 256);
    } else {
        block_size = make_uint3(1, 1, 512);
    }
    if (rhs_matrix_size.x < lhs_matrix_size.y) {
        luisa::swap(block_size.x, block_size.y);
    }
    block_size = block_size.zyx();
    uint2 size = make_uint2(rhs_matrix_size.x, lhs_matrix_size.y);
    uint2 lhs_size = make_uint2(lhs_matrix_size.x, size.y);
    uint2 rhs_size = make_uint2(size.x, rhs_matrix_size.y);
    auto kernel = Kernel3D([=](BufferVar<T> lhs, BufferVar<T> rhs, BufferVar<T> output) {
        auto ReadTex = [&](BufferVar<T> &img, auto &&idx) {
            return img.read(idx);
        };
        auto WriteTex = [&](BufferVar<T> &img, auto &&idx, auto &&value) {
            img.write(idx, value);
        };
        set_block_size(block_size);
        // device
        UInt3 id = dispatch_id().zyx();
        Float r = 0.f;
        UInt lhs_global_offset = lhs_batch ? ((lhs_matrix_size.x * lhs_matrix_size.y) * id.z) : 0u;
        UInt rhs_gloabal_offset = rhs_batch ? ((rhs_matrix_size.x * rhs_matrix_size.y) * id.z) : 0u;
        UInt output_global_offset = (size.x * size.y) * id.z;
        for (auto i : dynamic_range(iterate_size)) {
            auto lhs_val = ReadTex(lhs, lhs_global_offset + lhs_size.y * i + id.y);
            auto rhs_val = ReadTex(rhs, rhs_gloabal_offset + rhs_size.y * id.x + i);
            r += Float(lhs_val * rhs_val);
        };
        // bias
        if (lhs_matrix_size.x < rhs_matrix_size.y) {
            for (auto i : dynamic_range(iterate_size, rhs_matrix_size.y)) {
                r += Float(ReadTex(rhs, rhs_gloabal_offset + rhs_size.y * id.x + i));
            }
        } else if (lhs_matrix_size.x > rhs_matrix_size.y) {
            for (auto i : dynamic_range(iterate_size, lhs_matrix_size.x)) {
                r += Float(ReadTex(lhs, lhs_global_offset + lhs_size.y * i + id.y));
            }
        }
        // TDOO: fused activation
        WriteTex(output,
                 output_global_offset + size.y * id.x + id.y,
                 VarType(r));
    });
    return {kernel, make_uint3(batch_size, size.yx())};
}
template<typename T>
auto create_tensor(Device &device, size_t element_size) {
    return device.create_buffer<T>(element_size);
}

int main(int argc, char *argv[]) {
    auto ctx = Context(argv[0]);
    auto device = ctx.create_device("dx");
    auto stream = device.create_stream();

    size_t batch_size = 1920ull * 1080ull;
    size_t node_size = 4;
    auto a_buffer = create_tensor<half>(device, batch_size * node_size * node_size);
    auto b_buffer = create_tensor<half>(device, node_size);
    auto r_buffer = create_tensor<half>(device, batch_size * node_size);

    auto kernel_pack = get_matrix_multiple_kernel<half>(uint2(node_size, node_size), uint2(1, node_size), batch_size, true, false);
    auto shader = device.compile(kernel_pack.kernel);
    Clock clk;
    LUISA_INFO("Dispatch size {}", kernel_pack.dispatch_size);
    stream
        << shader(a_buffer, b_buffer, r_buffer).dispatch(kernel_pack.dispatch_size)
        << synchronize();
    CommandList cmdlist;
    for (int i = 0; i < 5; ++i) {
        cmdlist << shader(a_buffer, b_buffer, r_buffer).dispatch(kernel_pack.dispatch_size);
    }
    clk.tic();
    stream << cmdlist.commit() << synchronize();
    auto time = clk.toc();
    LUISA_INFO("Final time {}", time);
}