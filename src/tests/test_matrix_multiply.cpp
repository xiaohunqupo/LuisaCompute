#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/clock.h>
#include <fstream>
using namespace luisa;
using namespace luisa::compute;
struct DispatchPack {
    Kernel3D<void(Buffer<float>, Buffer<float>, Buffer<float>)> kernel;
    uint3 dispatch_size;
};

DispatchPack get_matrix_multiple_kernel(uint2 lhs_matrix_size, uint2 rhs_matrix_size, uint batch_size, bool lhs_batch, bool rhs_batch) {
    LUISA_ASSERT(lhs_matrix_size.x == rhs_matrix_size.y);
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
    uint3 size = make_uint3(rhs_matrix_size.x, lhs_matrix_size.y, batch_size);
    uint2 lhs_size = make_uint2(lhs_matrix_size.x, size.y);
    uint2 rhs_size = make_uint2(size.x, lhs_matrix_size.x);
    auto kernel = Kernel3D([=](BufferVar<float> lhs, BufferVar<float> rhs, BufferVar<float> output) {
        set_block_size(block_size);
        // device
        UInt3 id = dispatch_id().zyx();
        Float r = 0.f;
        UInt lhs_global_offset = lhs_batch ? ((lhs_matrix_size.x * lhs_matrix_size.y) * id.z) : 0u;
        UInt rhs_gloabal_offset = rhs_batch ? ((rhs_matrix_size.x * rhs_matrix_size.y) * id.z) : 0u;
        UInt output_global_offset = (size.x * size.y) * id.z;
        auto compute_code = [&](auto i) {
            auto lhs_val = lhs.read(lhs_global_offset + lhs_size.y * i + id.y);
            auto rhs_val = rhs.read(rhs_gloabal_offset + rhs_size.y * id.x + i);
            r += lhs_val * rhs_val;
        };
        $for (i, lhs_size.x) {
            compute_code(i);
        };
        // TDOO: fused activation
        output.write(
            output_global_offset + size.y * id.x + id.y,
            r);
    });
    return {kernel, size.zyx()};
}

int main(int argc, char *argv[]) {
    auto ctx = Context(argv[0]);
    auto device = ctx.create_device("dx");
    auto stream = device.create_stream();

    size_t batch_size = 1920ull * 1080ull;
    Buffer<float> a_buffer = device.create_buffer<float>(batch_size * 16ull * 16ull);
    Buffer<float> b_buffer = device.create_buffer<float>(16ull);
    Buffer<float> r_buffer = device.create_buffer<float>(batch_size * 16ull);
    auto kernel_pack = get_matrix_multiple_kernel(uint2(16, 16), uint2(1, 16), batch_size, true, false);
    auto shader = device.compile(kernel_pack.kernel);
    Clock clk;
    LUISA_INFO("Dispatch size {}", kernel_pack.dispatch_size);
    stream
        << shader(a_buffer, b_buffer, r_buffer).dispatch(kernel_pack.dispatch_size)
        << synchronize();
    CommandList cmdlist;
    cmdlist << shader(a_buffer, b_buffer, r_buffer).dispatch(kernel_pack.dispatch_size);
    clk.tic();
    stream << cmdlist.commit() << synchronize();
    auto time = clk.toc();
    LUISA_INFO("Final time {}", time);
}