//
// Created by mike on 11/1/25.
//

#include "cuda_codegen_llvm_impl.h"

#include <llvm/IR/Mangler.h>

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_translate_resource_query_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceQueryInst *inst) noexcept {
    switch (inst->op()) {
        case xir::ResourceQueryOp::BUFFER_SIZE: break;
        case xir::ResourceQueryOp::BYTE_BUFFER_SIZE: break;
        case xir::ResourceQueryOp::TEXTURE2D_SIZE: break;
        case xir::ResourceQueryOp::TEXTURE3D_SIZE: break;
        case xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE: break;
        case xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: break;
        case xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS: break;
        case xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS: break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM: break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID: break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: break;
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: break;
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: break;
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL: break;
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT: break;
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: break;
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: break;
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: break;
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: break;
    }
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *CUDACodegenLLVMImpl::_translate_resource_read_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceReadInst *inst) noexcept {
    switch (inst->op()) {
        case xir::ResourceReadOp::BUFFER_READ: break;
        case xir::ResourceReadOp::BUFFER_VOLATILE_READ: break;
        case xir::ResourceReadOp::BYTE_BUFFER_READ: break;
        case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: break;
        case xir::ResourceReadOp::TEXTURE2D_READ: break;
        case xir::ResourceReadOp::TEXTURE3D_READ: break;
        case xir::ResourceReadOp::BINDLESS_BUFFER_READ: break;
        case xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ: break;
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ: break;
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ: break;
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL: break;
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: break;
        case xir::ResourceReadOp::DEVICE_ADDRESS_READ: break;
    }
    LUISA_NOT_IMPLEMENTED();
}

void CUDACodegenLLVMImpl::_translate_resource_write_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceWriteInst *inst) noexcept {
    switch (inst->op()) {
        case xir::ResourceWriteOp::BUFFER_WRITE: break;
        case xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE: break;
        case xir::ResourceWriteOp::BYTE_BUFFER_WRITE: break;
        case xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: break;
        case xir::ResourceWriteOp::TEXTURE2D_WRITE: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, 0);
            auto llvm_texture_storage = b.CreateExtractValue(llvm_texture, 1);
            auto llvm_func = _get_texture2d_write_function(llvm::cast<llvm::VectorType>(llvm_value->getType()));
            b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_coord, llvm_value});
            return;
        }
        case xir::ResourceWriteOp::TEXTURE3D_WRITE: break;
        case xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE: break;
        case xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE: break;
        case xir::ResourceWriteOp::DEVICE_ADDRESS_WRITE: break;
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: break;
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK: break;
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY: break;
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID: break;
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX: break;
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: break;
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL: break;
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT: break;
    }
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::cuda
