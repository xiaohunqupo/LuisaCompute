//
// Created by mike on 11/1/25.
//

#include "cuda_codegen_llvm_impl.h"

#include <llvm/IR/Mangler.h>

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_translate_resource_query_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceQueryInst *inst) noexcept {
    switch (inst->op()) {
        case xir::ResourceQueryOp::BUFFER_SIZE: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_size_bytes = b.CreateExtractValue(llvm_buffer, 1);
            auto elem_type = inst->operand(0)->type()->element();
            auto llvm_size_elements = b.CreateUDiv(llvm_size_bytes, b.getInt64(elem_type->size()));
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreateZExtOrTrunc(llvm_size_elements, llvm_result_type);
        }
        case xir::ResourceQueryOp::BYTE_BUFFER_SIZE: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_size = b.CreateExtractValue(llvm_buffer, 1);
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreateZExtOrTrunc(llvm_size, llvm_result_type);
        }
        case xir::ResourceQueryOp::TEXTURE2D_SIZE: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::int2>() || inst->type() == Type::of<luisa::uint2>());
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_handle = b.CreateExtractValue(llvm_texture, 0);
            auto llvm_width = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_width, {llvm_handle});
            auto llvm_height = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_height, {llvm_handle});
            return _create_llvm_vector(b, {llvm_width, llvm_height});
        }
        case xir::ResourceQueryOp::TEXTURE3D_SIZE: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::int3>() || inst->type() == Type::of<luisa::uint3>());
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_handle = b.CreateExtractValue(llvm_texture, 0);
            auto llvm_width = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_width, {llvm_handle});
            auto llvm_height = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_height, {llvm_handle});
            auto llvm_depth = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_depth, {llvm_handle});
            return _create_llvm_vector(b, {llvm_width, llvm_height, llvm_depth});
        }
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
        case xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            return b.CreateExtractValue(llvm_buffer, 0);
        }
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
    switch (auto op = inst->op()) {
        case xir::ResourceReadOp::BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BUFFER_VOLATILE_READ: {
            // create a fence before volatile read if volatile read is requested
            if (op == xir::ResourceReadOp::BUFFER_VOLATILE_READ) { b.CreateIntrinsic(llvm::Intrinsic::nvvm_membar_gl, {}); }
            // load from buffer with element index
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            LUISA_DEBUG_ASSERT(inst->type() == inst->operand(0)->type()->element());
            auto elem_type = inst->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_index, elem_type->size(), elem_type->size());
            return _load_llvm_value(b, llvm_elem_ptr, elem_type);
        }
        case xir::ResourceReadOp::BYTE_BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: {
            // create a fence before volatile read if volatile read is requested
            if (op == xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ) { b.CreateIntrinsic(llvm::Intrinsic::nvvm_membar_gl, {}); }
            // load from byte buffer with byte offset
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_byte_offset = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto elem_type = inst->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_byte_offset, 1, elem_type->size());
            return _load_llvm_value(b, llvm_elem_ptr, elem_type);
        }
        case xir::ResourceReadOp::TEXTURE2D_READ: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, 0);
            auto llvm_texture_storage = b.CreateExtractValue(llvm_texture, 1);
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            LUISA_DEBUG_ASSERT(llvm_result_type->isVectorTy());
            auto llvm_func = _get_texture2d_read_function(llvm::cast<llvm::VectorType>(llvm_result_type));
            return b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_coord});
        }
        case xir::ResourceReadOp::TEXTURE3D_READ: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, 0);
            auto llvm_texture_storage = b.CreateExtractValue(llvm_texture, 1);
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            LUISA_DEBUG_ASSERT(llvm_result_type->isVectorTy());
            auto llvm_func = _get_texture3d_read_function(llvm::cast<llvm::VectorType>(llvm_result_type));
            return b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_coord});
        }
        case xir::ResourceReadOp::BINDLESS_BUFFER_READ: break;
        case xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ: break;
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ: break;
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ: break;
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL: break;
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: break;
        case xir::ResourceReadOp::DEVICE_ADDRESS_READ: {
            auto llvm_address = b.CreateZExt(_get_llvm_value(b, func_ctx, inst->operand(0)), b.getInt64Ty(), "", true);
            auto llvm_ptr = b.CreateIntToPtr(llvm_address, b.getPtrTy());
            return _load_llvm_value(b, llvm_ptr, inst->type());
        }
    }
    LUISA_NOT_IMPLEMENTED();
}

void CUDACodegenLLVMImpl::_translate_resource_write_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceWriteInst *inst) noexcept {
    switch (auto op = inst->op()) {
        case xir::ResourceWriteOp::BUFFER_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            LUISA_DEBUG_ASSERT(inst->operand(2)->type() == inst->operand(0)->type()->element());
            auto elem_type = inst->operand(2)->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_index, elem_type->size(), elem_type->size());
            _store_llvm_value(b, llvm_elem_ptr, llvm_value, elem_type);
            // create a fence after volatile write if volatile write is requested
            if (op == xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE) { b.CreateIntrinsic(llvm::Intrinsic::nvvm_membar_gl, {}); }
            return;
        }
        case xir::ResourceWriteOp::BYTE_BUFFER_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_byte_offset = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            LUISA_DEBUG_ASSERT(inst->operand(2)->type() == inst->operand(0)->type()->element());
            auto elem_type = inst->operand(2)->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_byte_offset, 1, elem_type->size());
            _store_llvm_value(b, llvm_elem_ptr, llvm_value, elem_type);
            // create a fence after volatile write if volatile write is requested
            if (op == xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE) { b.CreateIntrinsic(llvm::Intrinsic::nvvm_membar_gl, {}); }
            return;
        }
        case xir::ResourceWriteOp::TEXTURE2D_WRITE: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, 0);
            auto llvm_texture_storage = b.CreateExtractValue(llvm_texture, 1);
            LUISA_DEBUG_ASSERT(llvm_value->getType()->isVectorTy());
            auto llvm_func = _get_texture2d_write_function(llvm::cast<llvm::VectorType>(llvm_value->getType()));
            b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_coord, llvm_value});
            return;
        }
        case xir::ResourceWriteOp::TEXTURE3D_WRITE: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, 0);
            auto llvm_texture_storage = b.CreateExtractValue(llvm_texture, 1);
            LUISA_DEBUG_ASSERT(llvm_value->getType()->isVectorTy());
            auto llvm_func = _get_texture3d_write_function(llvm::cast<llvm::VectorType>(llvm_value->getType()));
            b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_coord, llvm_value});
            return;
        }
        case xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE: break;
        case xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE: break;
        case xir::ResourceWriteOp::DEVICE_ADDRESS_WRITE: {
            auto llvm_address = b.CreateZExt(_get_llvm_value(b, func_ctx, inst->operand(0)), b.getInt64Ty(), "", true);
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_ptr = b.CreateIntToPtr(llvm_address, b.getPtrTy());
            _store_llvm_value(b, llvm_ptr, llvm_value, inst->operand(1)->type());
            return;
        }
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

llvm::Value *CUDACodegenLLVMImpl::_get_buffer_element_pointer(IB &b, llvm::Value *buffer, llvm::Value *index, size_t index_stride, size_t element_size) noexcept {
    auto buffer_data_ptr = b.CreateExtractValue(buffer, 0);
    auto buffer_size_bytes = b.CreateExtractValue(buffer, 1);
    auto size_type = buffer_size_bytes->getType();
    LUISA_DEBUG_ASSERT(size_type->isIntegerTy(64));
    index = b.CreateZExt(index, size_type, "", true);
    auto offset_bytes = index_stride == 1 ? index : b.CreateMul(index, b.getInt64(index_stride), "", true, true);
    // check oob if debug mode
    if (_config.enable_debug_info) {
        auto element_end = b.CreateAdd(offset_bytes, b.getInt64(element_size), "", true, true);
        auto inbounds = b.CreateICmpULE(element_end, buffer_size_bytes);
        _create_assertion_with_message(b, inbounds, "Buffer access out of bounds");
    }
    // get element pointer
    return b.CreateInBoundsGEP(b.getInt8Ty(), buffer_data_ptr, offset_bytes);
}

}// namespace luisa::compute::cuda
