#include <luisa/tensor/fallback/matmul_impl.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/vstl/md5.h>
namespace luisa::compute {
namespace gemm_detail {
template<typename T>
struct DispatchPack {
    Kernel3D<void(Buffer<T>, Buffer<T>, Buffer<T>)> kernel;
    uint3 dispatch_size;
};
struct GEMMKey {
    uint2 lhs_matrix_size;
    uint2 rhs_matrix_size;
    uint min_batch_size;
    uint batch;
    uint type;
    FusedActivation activation;
};
template<typename T>
DispatchPack<T> gemm_kernel(uint2 lhs_matrix_size, uint2 rhs_matrix_size, uint min_batch_size, bool lhs_batch, bool rhs_batch, FusedActivation activation) {
    using VarType = Var<T>;
    auto iterate_size = std::min(lhs_matrix_size.x, rhs_matrix_size.y);
    uint3 block_size;
    if (min_batch_size == 1) {
        block_size = make_uint3(8, 8, 1);
    } else if (min_batch_size < 4) {
        block_size = make_uint3(8, 4, min_batch_size);
    } else if (min_batch_size < 8) {
        block_size = make_uint3(4, 4, min_batch_size);
    } else if (min_batch_size < 16) {
        block_size = make_uint3(4, 2, min_batch_size);
    } else if (min_batch_size < 32) {
        block_size = make_uint3(2, 2, min_batch_size);
    } else if (min_batch_size < 64) {
        block_size = make_uint3(2, 1, min_batch_size);
    } else if (min_batch_size < 64ull * 65535ull) {
        block_size = make_uint3(1, 1, 64);
    } else if (min_batch_size < 128ull * 65535ull) {
        block_size = make_uint3(1, 1, 128);
    } else if (min_batch_size < 256ull * 65535ull) {
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
        auto ReadBuffer = [&](BufferVar<T> &img, auto &&idx) {
            return img.read(idx);
        };
        auto WriteBuffer = [&](BufferVar<T> &img, auto &&idx, auto &&value) {
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
            auto lhs_val = ReadBuffer(lhs, lhs_global_offset + lhs_size.y * i + id.y);
            auto rhs_val = ReadBuffer(rhs, rhs_gloabal_offset + rhs_size.y * id.x + i);
            r += Float(lhs_val * rhs_val);
        };
        // bias
        if (lhs_matrix_size.x < rhs_matrix_size.y) {
            for (auto i : dynamic_range(iterate_size, rhs_matrix_size.y)) {
                r += Float(ReadBuffer(rhs, rhs_gloabal_offset + rhs_size.y * id.x + i));
            }
        } else if (lhs_matrix_size.x > rhs_matrix_size.y) {
            for (auto i : dynamic_range(iterate_size, lhs_matrix_size.x)) {
                r += Float(ReadBuffer(lhs, lhs_global_offset + lhs_size.y * i + id.y));
            }
        }
        // TDOO: fused activation
        WriteBuffer(output,
                    output_global_offset + size.y * id.x + id.y,
                    VarType(r));
    });
    return {kernel, make_uint3(min_batch_size, size.yx())};
}
}// namespace gemm_detail
MatMulImpl::MatMulImpl(
    DeviceInterface *device,
    ShaderManager *shader_manager,
    GEMMExpr *expr)
    : expr(expr) {
    auto set_disp_pack = [&](auto &disp_pack) {
        dispatch_size = disp_pack.desired_dispatch_size;
        uniform_size = disp_pack.uniform_size;
        shader_handle = disp_pack.shader_handle;
    };
    uint2 lhs_matrix_size = uint2(expr->lhs_tensor->get_size(0), expr->lhs_tensor->get_size(1));
    uint2 rhs_matrix_size = uint2(expr->rhs_tensor->get_size(0), expr->rhs_tensor->get_size(1));
    uint min_batch_size = expr->output_tensor->get_size(2);
    uint lhs_batch = expr->lhs_tensor->get_size(2);
    uint rhs_batch = expr->rhs_tensor->get_size(2);
    gemm_detail::GEMMKey key{
        .lhs_matrix_size = lhs_matrix_size,
        .rhs_matrix_size = rhs_matrix_size,
        .min_batch_size = min_batch_size,
        .batch = uint(lhs_batch ? 1 : 0) | uint(rhs_batch ? 2 : 0),
        .activation = expr->fused_activation};
    switch (expr->lhs_tensor->element_type()) {
        case TensorElementType::Float16: {
            key.type = 0;
            auto disp_pack = shader_manager->add_shader(
                TensorExpr::Tag::EGEMMExpr,
                vstd::MD5{{reinterpret_cast<uint8_t const *>(&key), sizeof(key)}},
                [&]() {
                    auto disp_pack = gemm_detail::gemm_kernel<half>(lhs_matrix_size, rhs_matrix_size, min_batch_size, lhs_batch, rhs_batch, expr->fused_activation);
                    auto create_info = device->create_shader(ShaderOption{}, Function{disp_pack.kernel.function().get()});
                    return ShaderManager::ShaderDispatch{
                        create_info.handle,
                        disp_pack.dispatch_size,
                        ShaderDispatchCmdEncoder::compute_uniform_size(disp_pack.kernel.function()->unbound_arguments())};
                });
            set_disp_pack(disp_pack);
        } break;
        case TensorElementType::Float32: {
            key.type = 1;
            auto disp_pack = shader_manager->add_shader(
                TensorExpr::Tag::EGEMMExpr,
                vstd::MD5{{reinterpret_cast<uint8_t const *>(&key), sizeof(key)}},
                [&]() {
                    auto disp_pack = gemm_detail::gemm_kernel<float>(lhs_matrix_size, rhs_matrix_size, min_batch_size, lhs_batch, rhs_batch, expr->fused_activation);
                    auto create_info = device->create_shader(ShaderOption{}, Function{disp_pack.kernel.function().get()});
                    return ShaderManager::ShaderDispatch{
                        create_info.handle,
                        disp_pack.dispatch_size,
                        ShaderDispatchCmdEncoder::compute_uniform_size(disp_pack.kernel.function()->unbound_arguments())};
                });
            set_disp_pack(disp_pack);
        } break;
        default: {
            LUISA_ERROR("Only float 16 and float 32 supported.");
        } break;
    }
}
MatMulImpl::~MatMulImpl() {
}
void MatMulImpl::execute(FallbackTensorCallback *callback, CommandList &cmdlist) const {
    auto lhs_buffer = callback->get_tensor_buffer(expr->lhs_tensor);
    auto rhs_buffer = callback->get_tensor_buffer(expr->rhs_tensor);
    auto output_buffer = callback->get_tensor_buffer(expr->output_tensor);
    ComputeDispatchCmdEncoder dispatcher{
        shader_handle,
        3,
        uniform_size};
    dispatcher.encode_buffer(lhs_buffer.handle, lhs_buffer.offset, lhs_buffer.size);
    dispatcher.encode_buffer(rhs_buffer.handle, rhs_buffer.offset, rhs_buffer.size);
    dispatcher.encode_buffer(output_buffer.handle, output_buffer.offset, output_buffer.size);
    dispatcher.set_dispatch_size(dispatch_size);
    cmdlist << std::move(dispatcher).build();
}
}// namespace luisa::compute