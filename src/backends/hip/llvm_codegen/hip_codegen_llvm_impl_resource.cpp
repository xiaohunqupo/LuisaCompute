//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_translate_resource_query_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceQueryInst *inst) noexcept {
    switch (auto op = inst->op()) {
        case xir::ResourceQueryOp::BUFFER_SIZE: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_size_bytes = b.CreateExtractValue(llvm_buffer, llvm_buffer_type_size_index);
            auto elem_type = inst->operand(0)->type()->element();
            auto llvm_size_elements = b.CreateUDiv(llvm_size_bytes, b.getInt64(elem_type->size()));
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreateZExtOrTrunc(llvm_size_elements, llvm_result_type);
        }
        case xir::ResourceQueryOp::BYTE_BUFFER_SIZE: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_size = b.CreateExtractValue(llvm_buffer, llvm_buffer_type_size_index);
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreateZExtOrTrunc(llvm_size, llvm_result_type);
        }
        case xir::ResourceQueryOp::TEXTURE2D_SIZE: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::int2>() || inst->type() == Type::of<luisa::uint2>());
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_width = static_cast<llvm::Value *>(b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_getresinfo_2d, {b.getFloatTy()}, {llvm_handle}));
            auto llvm_height = b.CreateExtractValue(llvm_width, 1);
            llvm_width = b.CreateExtractValue(llvm_width, 0);
            return _create_llvm_vector(b, {llvm_width, llvm_height});
        }
        case xir::ResourceQueryOp::TEXTURE3D_SIZE: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::int3>() || inst->type() == Type::of<luisa::uint3>());
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_resinfo = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_getresinfo_3d, {b.getFloatTy()}, {llvm_handle});
            auto llvm_width = b.CreateExtractValue(llvm_resinfo, 0);
            auto llvm_height = b.CreateExtractValue(llvm_resinfo, 1);
            auto llvm_depth = b.CreateExtractValue(llvm_resinfo, 2);
            return _create_llvm_vector(b, {llvm_width, llvm_height, llvm_depth});
        }
        case xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_slot_ptr = _get_bindless_array_slot_pointer(b, llvm_bindless_array, llvm_index);
            auto llvm_slot_type = _get_llvm_bindless_array_slot_type();
            auto llvm_buffer_size_ptr = b.CreateStructGEP(llvm_slot_type, llvm_slot_ptr, llvm_bindless_array_slot_type_buffer_size_index);
            auto llvm_buffer_size = static_cast<llvm::Value *>(b.CreateLoad(
                llvm_slot_type->getStructElementType(llvm_bindless_array_slot_type_buffer_size_index), llvm_buffer_size_ptr));
            if (op == xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE) {
                auto elem_stride = b.CreateZExt(_get_llvm_value(b, func_ctx, inst->operand(2)), llvm_buffer_size->getType());
                llvm_buffer_size = b.CreateUDiv(llvm_buffer_size, elem_stride);
            }
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreateZExtOrTrunc(llvm_buffer_size, llvm_result_type);
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::int2>() || inst->type() == Type::of<luisa::uint2>());
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 2);
            auto llvm_resinfo = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_getresinfo_2d, {b.getFloatTy()}, {llvm_handle});
            auto llvm_width = b.CreateExtractValue(llvm_resinfo, 0);
            auto llvm_height = b.CreateExtractValue(llvm_resinfo, 1);
            auto llvm_size = _create_llvm_vector(b, {llvm_width, llvm_height});
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL) {
                auto llvm_level = b.CreateVectorSplat(2, _get_llvm_value(b, func_ctx, inst->operand(2)));
                llvm_size = b.CreateLShr(llvm_size, llvm_level);
            }
            return llvm_size;
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::int3>() || inst->type() == Type::of<luisa::uint3>());
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 3);
            auto llvm_resinfo = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_getresinfo_3d, {b.getFloatTy()}, {llvm_handle});
            auto llvm_width = b.CreateExtractValue(llvm_resinfo, 0);
            auto llvm_height = b.CreateExtractValue(llvm_resinfo, 1);
            auto llvm_depth = b.CreateExtractValue(llvm_resinfo, 2);
            auto llvm_size = _create_llvm_vector(b, {llvm_width, llvm_height, llvm_depth});
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL) {
                auto llvm_level = b.CreateVectorSplat(3, _get_llvm_value(b, func_ctx, inst->operand(2)));
                llvm_size = b.CreateLShr(llvm_size, llvm_level);
            }
            return llvm_size;
        }
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 2);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_result = static_cast<llvm::Value *>(nullptr);
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE) {
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_2d,
                                                {b.getFloatTy(), b.getFloatTy(), b.getFloatTy()},
                                                {llvm_handle, llvm_coord_x, llvm_coord_y});
            } else if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL) {
                auto llvm_level = _get_llvm_value(b, func_ctx, inst->operand(3));
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_l_2d,
                                                {b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy()},
                                                {llvm_handle, llvm_coord_x, llvm_coord_y, llvm_level});
            } else {
                if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL) {
                    LUISA_WARNING_WITH_LOCATION("Level parameter in BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL is ignored in HIP backend.");
                }
                auto llvm_ddx = _get_llvm_value(b, func_ctx, inst->operand(3));
                auto llvm_ddy = _get_llvm_value(b, func_ctx, inst->operand(4));
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_2d,
                                                {b.getFloatTy(), b.getFloatTy(), b.getFloatTy()},
                                                {llvm_handle, llvm_coord_x, llvm_coord_y});
            }
            auto llvm_result_x = b.CreateExtractValue(llvm_result, 0);
            auto llvm_result_y = b.CreateExtractValue(llvm_result, 1);
            auto llvm_result_z = b.CreateExtractValue(llvm_result, 2);
            auto llvm_result_w = b.CreateExtractValue(llvm_result, 3);
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 3);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_coord_z = b.CreateExtractElement(llvm_coord, b.getInt64(2));
            auto llvm_result = static_cast<llvm::Value *>(nullptr);
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE) {
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_3d,
                                                {b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy()},
                                                {llvm_handle, llvm_coord_x, llvm_coord_y, llvm_coord_z});
            } else if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL) {
                auto llvm_level = _get_llvm_value(b, func_ctx, inst->operand(3));
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_l_3d,
                                                {b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy()},
                                                {llvm_handle, llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_level});
            } else {
                if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL) {
                    LUISA_WARNING_WITH_LOCATION("Level parameter in BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL is ignored in HIP backend.");
                }
                auto llvm_ddx = _get_llvm_value(b, func_ctx, inst->operand(3));
                auto llvm_ddy = _get_llvm_value(b, func_ctx, inst->operand(4));
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_3d,
                                                {b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy()},
                                                {llvm_handle, llvm_coord_x, llvm_coord_y, llvm_coord_z});
            }
            auto llvm_result_x = b.CreateExtractValue(llvm_result, 0);
            auto llvm_result_y = b.CreateExtractValue(llvm_result, 1);
            auto llvm_result_z = b.CreateExtractValue(llvm_result, 2);
            auto llvm_result_w = b.CreateExtractValue(llvm_result, 3);
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
        }
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
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreatePtrToInt(b.CreateExtractValue(llvm_buffer, llvm_buffer_type_ptr_index), llvm_result_type);
        }
        case xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_slot_ptr = _get_bindless_array_slot_pointer(b, llvm_bindless_array, llvm_index);
            auto llvm_slot_type = _get_llvm_bindless_array_slot_type();
            auto llvm_buffer_ptr = b.CreateLoad(llvm_slot_type->getStructElementType(llvm_bindless_array_slot_type_buffer_ptr_index), llvm_slot_ptr);
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreatePtrToInt(llvm_buffer_ptr, llvm_result_type);
        }
        default: LUISA_NOT_IMPLEMENTED();
    }
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_resource_read_inst(IB &b, const FunctionContext &func_ctx, const xir::ResourceReadInst *inst) noexcept {
    switch (auto op = inst->op()) {
        case xir::ResourceReadOp::BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BUFFER_VOLATILE_READ: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            LUISA_DEBUG_ASSERT(inst->type() == inst->operand(0)->type()->element());
            auto elem_type = inst->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_index, elem_type->size(), elem_type->size());
            return _load_llvm_value(b, llvm_elem_ptr, elem_type);
        }
        case xir::ResourceReadOp::BYTE_BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_byte_offset = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto elem_type = inst->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_byte_offset, 1, elem_type->size());
            return _load_llvm_value(b, llvm_elem_ptr, elem_type);
        }
        case xir::ResourceReadOp::TEXTURE2D_READ: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_texture_storage = b.CreateExtractValue(llvm_texture, llvm_texture_type_storage_index);
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            LUISA_DEBUG_ASSERT(llvm_result_type->isVectorTy());
            auto llvm_func = _get_texture2d_read_function(llvm::cast<llvm::VectorType>(llvm_result_type));
            return b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_coord});
        }
        case xir::ResourceReadOp::TEXTURE3D_READ: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_texture_storage = b.CreateExtractValue(llvm_texture, llvm_texture_type_storage_index);
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            LUISA_DEBUG_ASSERT(llvm_result_type->isVectorTy());
            auto llvm_func = _get_texture3d_read_function(llvm::cast<llvm::VectorType>(llvm_result_type));
            return b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_coord});
        }
        case xir::ResourceReadOp::BINDLESS_BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ: {
            auto llvm_bindless = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_slot_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_slot_ptr = _get_bindless_array_slot_pointer(b, llvm_bindless, llvm_slot_index);
            auto llvm_buffer_type = _get_llvm_buffer_type();
            LUISA_DEBUG_ASSERT(llvm_buffer_type->getStructNumElements() == 2 &&
                               llvm_buffer_type->getStructElementType(llvm_buffer_type_ptr_index) ==
                                   _get_llvm_bindless_array_slot_type()->getStructElementType(llvm_buffer_type_ptr_index) &&
                               llvm_buffer_type->getStructElementType(llvm_buffer_type_size_index) ==
                                   _get_llvm_bindless_array_slot_type()->getStructElementType(llvm_buffer_type_size_index));
            auto llvm_buffer = b.CreateLoad(llvm_buffer_type, llvm_slot_ptr);
            auto llvm_index_or_offset = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto elem_type = inst->type();
            auto index_stride = (op == xir::ResourceReadOp::BINDLESS_BUFFER_READ) ? elem_type->size() : 1;
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_index_or_offset, index_stride, elem_type->size());
            return _load_llvm_value(b, llvm_elem_ptr, elem_type);
        }
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 2);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_load_2d,
                                                 {b.getFloatTy(), b.getInt32Ty(), b.getInt32Ty()},
                                                 {llvm_handle, llvm_coord_x, llvm_coord_y});
            auto llvm_result_x = b.CreateSIToFP(b.CreateExtractValue(llvm_result, 0), b.getFloatTy());
            auto llvm_result_y = b.CreateSIToFP(b.CreateExtractValue(llvm_result, 1), b.getFloatTy());
            auto llvm_result_z = b.CreateSIToFP(b.CreateExtractValue(llvm_result, 2), b.getFloatTy());
            auto llvm_result_w = b.CreateSIToFP(b.CreateExtractValue(llvm_result, 3), b.getFloatTy());
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
        }
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 3);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_coord_z = b.CreateExtractElement(llvm_coord, b.getInt64(2));
            auto llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_load_3d,
                                                 {b.getFloatTy(), b.getInt32Ty(), b.getInt32Ty(), b.getInt32Ty()},
                                                 {llvm_handle, llvm_coord_x, llvm_coord_y, llvm_coord_z});
            auto llvm_result_x = b.CreateSIToFP(b.CreateExtractValue(llvm_result, 0), b.getFloatTy());
            auto llvm_result_y = b.CreateSIToFP(b.CreateExtractValue(llvm_result, 1), b.getFloatTy());
            auto llvm_result_z = b.CreateSIToFP(b.CreateExtractValue(llvm_result, 2), b.getFloatTy());
            auto llvm_result_w = b.CreateSIToFP(b.CreateExtractValue(llvm_result, 3), b.getFloatTy());
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
        }
        case xir::ResourceReadOp::DEVICE_ADDRESS_READ: {
            auto llvm_address = b.CreateZExt(_get_llvm_value(b, func_ctx, inst->operand(0)), b.getInt64Ty(), "", true);
            auto llvm_ptr = b.CreateIntToPtr(llvm_address, b.getPtrTy());
            return _load_llvm_value(b, llvm_ptr, inst->type());
        }
    }
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_resource_write_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceWriteInst *inst) noexcept {
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
            return;
        }
        case xir::ResourceWriteOp::BYTE_BUFFER_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_byte_offset = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto elem_type = inst->operand(2)->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_byte_offset, 1, elem_type->size());
            _store_llvm_value(b, llvm_elem_ptr, llvm_value, elem_type);
            return;
        }
        case xir::ResourceWriteOp::TEXTURE2D_WRITE: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_texture_storage = b.CreateExtractValue(llvm_texture, llvm_texture_type_storage_index);
            LUISA_DEBUG_ASSERT(llvm_value->getType()->isVectorTy());
            auto llvm_func = _get_texture2d_write_function(llvm::cast<llvm::VectorType>(llvm_value->getType()));
            b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_coord, llvm_value});
            return;
        }
        case xir::ResourceWriteOp::TEXTURE3D_WRITE: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_texture_storage = b.CreateExtractValue(llvm_texture, llvm_texture_type_storage_index);
            LUISA_DEBUG_ASSERT(llvm_value->getType()->isVectorTy());
            auto llvm_func = _get_texture3d_write_function(llvm::cast<llvm::VectorType>(llvm_value->getType()));
            b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_coord, llvm_value});
            return;
        }
        case xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE: {
            auto llvm_bindless = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_slot_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_slot_ptr = _get_bindless_array_slot_pointer(b, llvm_bindless, llvm_slot_index);
            auto llvm_buffer_type = _get_llvm_buffer_type();
            LUISA_DEBUG_ASSERT(llvm_buffer_type->getStructNumElements() == 2 &&
                               llvm_buffer_type->getStructElementType(llvm_buffer_type_ptr_index) ==
                                   _get_llvm_bindless_array_slot_type()->getStructElementType(llvm_buffer_type_ptr_index) &&
                               llvm_buffer_type->getStructElementType(llvm_buffer_type_size_index) ==
                                   _get_llvm_bindless_array_slot_type()->getStructElementType(llvm_buffer_type_size_index));
            auto llvm_buffer = b.CreateLoad(llvm_buffer_type, llvm_slot_ptr);
            auto llvm_index_or_offset = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto value = inst->operand(3);
            auto elem_type = value->type();
            auto index_stride = (op == xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE) ? elem_type->size() : 1;
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_index_or_offset, index_stride, elem_type->size());
            auto llvm_value = _get_llvm_value(b, func_ctx, value);
            return _store_llvm_value(b, llvm_elem_ptr, llvm_value, elem_type);
        }
        case xir::ResourceWriteOp::DEVICE_ADDRESS_WRITE: {
            auto llvm_address = b.CreateZExt(_get_llvm_value(b, func_ctx, inst->operand(0)), b.getInt64Ty(), "", true);
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_ptr = b.CreateIntToPtr(llvm_address, b.getPtrTy());
            _store_llvm_value(b, llvm_ptr, llvm_value, inst->operand(1)->type());
            return;
        }
    }
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_get_buffer_element_pointer(IB &b, llvm::Value *buffer, llvm::Value *index, size_t index_stride, size_t element_size) noexcept {
    auto buffer_data_ptr = b.CreateExtractValue(buffer, llvm_buffer_type_ptr_index);
    auto buffer_size_bytes = b.CreateExtractValue(buffer, llvm_buffer_type_size_index);
    auto size_type = buffer_size_bytes->getType();
    LUISA_DEBUG_ASSERT(size_type->isIntegerTy(64));
    index = b.CreateZExt(index, size_type, "", true);
    auto offset_bytes = index_stride == 1 ? index : b.CreateMul(index, b.getInt64(index_stride), "", true, true);
    return b.CreateInBoundsGEP(b.getInt8Ty(), buffer_data_ptr, offset_bytes);
}

llvm::Value *HIPCodegenLLVMImpl::_get_bindless_array_slot_pointer(IB &b, llvm::Value *bindless_array, llvm::Value *slot_index) noexcept {
    auto slots = b.CreateExtractValue(bindless_array, llvm_bindless_array_type_slots_index);
    auto slot_count = b.CreateExtractValue(bindless_array, llvm_bindless_array_type_size_index);
    slot_index = b.CreateZExt(slot_index, slot_count->getType(), "", true);
    auto slot_type = _get_llvm_bindless_array_slot_type();
    return b.CreateInBoundsGEP(slot_type, slots, slot_index);
}

llvm::Value *HIPCodegenLLVMImpl::_get_bindless_array_texture_handle(IB &b, llvm::Value *bindless_array, llvm::Value *slot_index, int dim) noexcept {
    auto slot_ptr = _get_bindless_array_slot_pointer(b, bindless_array, slot_index);
    auto slot_type = _get_llvm_bindless_array_slot_type();
    auto i = dim == 2 ? llvm_bindless_array_slot_type_texture2d_handle_index :
                        llvm_bindless_array_slot_type_texture3d_handle_index;
    auto handle_ptr = b.CreateStructGEP(slot_type, slot_ptr, i);
    return b.CreateLoad(slot_type->getStructElementType(i), handle_ptr);
}

llvm::Value *HIPCodegenLLVMImpl::_get_accel_instance_pointer(IB &b, llvm::Value *accel, llvm::Value *instance_index) noexcept {
    auto instances = b.CreateExtractValue(accel, llvm_accel_type_instances_index);
    instance_index = b.CreateZExt(instance_index, b.getInt64Ty(), "", true);
    return b.CreateInBoundsGEP(_get_llvm_accel_instance_type(), instances, instance_index);
}

llvm::Value *HIPCodegenLLVMImpl::_load_accel_affine_matrix(IB &b, llvm::Value *affine_ptr) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_store_accel_affine_matrix(IB &b, llvm::Value *affine_ptr, llvm::Value *matrix) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::hip
