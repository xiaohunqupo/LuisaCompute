//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

size_t HIPCodegenLLVMImpl::_get_type_alignment(const Type *type) noexcept {
    if (type->is_basic() || type->is_array() || type->is_structure()) {
        return type->alignment();
    }
    return 16;
}

const HIPCodegenLLVMImpl::LLVMTypeInfo *HIPCodegenLLVMImpl::_get_llvm_type(const Type *type) noexcept {
    if (auto iter = _xir_to_llvm_type.find(type); iter != _xir_to_llvm_type.end()) {
        return iter->second.get();
    }
    auto llvm_type_info = [this, type]() noexcept -> luisa::unique_ptr<LLVMTypeInfo> {
        auto make_info = [this](llvm::Type *mem_t, llvm::Type *reg_t, size_t s, size_t a,
                                luisa::vector<size_t> member_indices = {},
                                luisa::vector<size_t> member_offsets = {}) noexcept {
            auto info = luisa::make_unique<LLVMTypeInfo>();
            info->mem_type = mem_t;
            info->reg_type = reg_t;
            info->member_indices = std::move(member_indices);
            info->member_offsets = std::move(member_offsets);
            return info;
        };
        switch (type->tag()) {
            case Type::Tag::BOOL: {
                auto t = llvm::Type::getInt1Ty(_llvm_context);
                return make_info(t, t, sizeof(bool), alignof(bool));
            }
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::UINT8: {
                auto t = llvm::Type::getInt8Ty(_llvm_context);
                return make_info(t, t, sizeof(int8_t), alignof(int8_t));
            }
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::UINT16: {
                auto t = llvm::Type::getInt16Ty(_llvm_context);
                return make_info(t, t, sizeof(int16_t), alignof(int16_t));
            }
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::UINT32: {
                auto t = llvm::Type::getInt32Ty(_llvm_context);
                return make_info(t, t, sizeof(int32_t), alignof(int32_t));
            }
            case Type::Tag::INT64: [[fallthrough]];
            case Type::Tag::UINT64: {
                auto t = llvm::Type::getInt64Ty(_llvm_context);
                return make_info(t, t, sizeof(int64_t), alignof(int64_t));
            }
            case Type::Tag::FLOAT16: {
                auto t = llvm::Type::getHalfTy(_llvm_context);
                return make_info(t, t, sizeof(luisa::half), alignof(luisa::half));
            }
            case Type::Tag::FLOAT32: {
                auto t = llvm::Type::getFloatTy(_llvm_context);
                return make_info(t, t, sizeof(float), alignof(float));
            }
            case Type::Tag::FLOAT64: {
                auto t = llvm::Type::getDoubleTy(_llvm_context);
                return make_info(t, t, sizeof(double), alignof(double));
            }
            case Type::Tag::VECTOR: {
                auto elem = _get_llvm_type(type->element());
                auto dim = type->dimension();
                auto llvm_reg_type = llvm::FixedVectorType::get(elem->reg_type, dim);
                auto llvm_mem_type = llvm::ArrayType::get(elem->mem_type, luisa::align(dim, 2));
                return make_info(llvm_mem_type, llvm_reg_type, type->size(), type->alignment());
            }
            case Type::Tag::MATRIX: {
                auto dim = type->dimension();
                auto col_type = Type::vector(type->element(), dim);
                auto col_llvm = _get_llvm_type(col_type);
                auto llvm_reg_type = llvm::ArrayType::get(col_llvm->reg_type, dim);
                auto llvm_mem_type = llvm::ArrayType::get(col_llvm->mem_type, dim);
                return make_info(llvm_mem_type, llvm_reg_type, type->size(), type->alignment());
            }
            case Type::Tag::ARRAY: {
                auto elem = _get_llvm_type(type->element());
                auto llvm_reg_type = llvm::ArrayType::get(elem->reg_type, type->dimension());
                auto llvm_mem_type = llvm::ArrayType::get(elem->mem_type, type->dimension());
                return make_info(llvm_mem_type, llvm_reg_type, type->size(), type->alignment());
            }
            case Type::Tag::STRUCTURE: {
                luisa::vector<size_t> member_indices;
                luisa::vector<size_t> member_offsets;
                llvm::SmallVector<llvm::Type *> llvm_member_reg_types;
                llvm::SmallVector<llvm::Type *> llvm_member_mem_types;
                auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
                auto current_offset = static_cast<size_t>(0u);
                for (auto member : type->members()) {
                    auto llvm_member_type = _get_llvm_type(member);
                    llvm_member_reg_types.emplace_back(llvm_member_type->reg_type);
                    auto next_offset = luisa::align(current_offset, member->alignment());
                    if (next_offset > current_offset) {
                        llvm_member_mem_types.emplace_back(
                            llvm::ArrayType::get(llvm_i8_type, next_offset - current_offset));
                    }
                    member_indices.emplace_back(llvm_member_mem_types.size());
                    member_offsets.emplace_back(next_offset);
                    llvm_member_mem_types.emplace_back(llvm_member_type->mem_type);
                    current_offset = next_offset + member->size();
                }
                if (current_offset < type->size()) {
                    llvm_member_mem_types.emplace_back(
                        llvm::ArrayType::get(llvm_i8_type, type->size() - current_offset));
                }
                auto llvm_reg_type = llvm::StructType::get(_llvm_context, llvm_member_reg_types, false);
                auto llvm_mem_type = llvm::StructType::get(_llvm_context, llvm_member_mem_types, false);
                return make_info(llvm_mem_type, llvm_reg_type, type->size(), type->alignment(),
                                 std::move(member_indices), std::move(member_offsets));
            }
            default: LUISA_ERROR_WITH_LOCATION("Unsupported type.");
        }
    }();
    auto [iter, _] = _xir_to_llvm_type.try_emplace(type, std::move(llvm_type_info));
    return iter->second.get();
}

const HIPCodegenLLVMImpl::KernelArgumentStruct *HIPCodegenLLVMImpl::_get_kernel_argument_struct(const xir::KernelFunction *func) noexcept {
    if (auto iter = _kernel_arg_struct_types.find(func); iter != _kernel_arg_struct_types.end()) {
        return iter->second.get();
    }
    LUISA_NOT_IMPLEMENTED();
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_buffer_type() noexcept {
    auto ptr_type = llvm::PointerType::get(_llvm_context, amdgpu_address_space_global);
    return llvm::StructType::create({ptr_type, llvm::Type::getInt64Ty(_llvm_context)}, "buffer");
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_texture_type() noexcept {
    return llvm::StructType::create({llvm::Type::getInt64Ty(_llvm_context), llvm::Type::getInt64Ty(_llvm_context)}, "texture");
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_bindless_array_type() noexcept {
    auto ptr_type = llvm::PointerType::get(_llvm_context, amdgpu_address_space_global);
    return llvm::StructType::create({ptr_type, llvm::Type::getInt64Ty(_llvm_context)}, "bindless_array");
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_bindless_array_slot_type() noexcept {
    return llvm::StructType::create({llvm::Type::getInt64Ty(_llvm_context),
                                     llvm::Type::getInt64Ty(_llvm_context),
                                     llvm::Type::getInt64Ty(_llvm_context),
                                     llvm::Type::getInt64Ty(_llvm_context)},
                                    "bindless_array_slot");
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_accel_type() noexcept {
    auto ptr_type = llvm::PointerType::get(_llvm_context, amdgpu_address_space_global);
    return llvm::StructType::create({llvm::Type::getInt64Ty(_llvm_context), ptr_type}, "accel");
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_accel_instance_type() noexcept {
    auto float4x3_type = llvm::ArrayType::get(llvm::FixedVectorType::get(llvm::Type::getFloatTy(_llvm_context), 4), 3);
    return llvm::StructType::create({float4x3_type,
                                     llvm::Type::getInt32Ty(_llvm_context),
                                     llvm::Type::getInt32Ty(_llvm_context),
                                     llvm::Type::getInt32Ty(_llvm_context),
                                     llvm::Type::getInt32Ty(_llvm_context),
                                     llvm::Type::getInt64Ty(_llvm_context)},
                                    "accel_instance");
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_ray_type() noexcept {
    return llvm::StructType::create({llvm::ArrayType::get(llvm::Type::getFloatTy(_llvm_context), 3),
                                     llvm::Type::getFloatTy(_llvm_context),
                                     llvm::ArrayType::get(llvm::Type::getFloatTy(_llvm_context), 3),
                                     llvm::Type::getFloatTy(_llvm_context)},
                                    "ray");
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_surface_hit_type() noexcept {
    return llvm::StructType::create({llvm::Type::getInt32Ty(_llvm_context),
                                     llvm::Type::getInt32Ty(_llvm_context),
                                     llvm::FixedVectorType::get(llvm::Type::getFloatTy(_llvm_context), 2),
                                     llvm::Type::getFloatTy(_llvm_context)},
                                    "surface_hit");
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_procedural_hit_type() noexcept {
    return llvm::StructType::create({llvm::Type::getInt32Ty(_llvm_context),
                                     llvm::Type::getInt32Ty(_llvm_context)},
                                    "procedural_hit");
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_committed_hit_type() noexcept {
    return llvm::StructType::create({llvm::Type::getInt32Ty(_llvm_context),
                                     llvm::Type::getInt32Ty(_llvm_context),
                                     llvm::FixedVectorType::get(llvm::Type::getFloatTy(_llvm_context), 2),
                                     llvm::Type::getInt32Ty(_llvm_context),
                                     llvm::Type::getFloatTy(_llvm_context)},
                                    "committed_hit");
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_ray_query_type() noexcept {
    return llvm::StructType::create({llvm::Type::getInt64Ty(_llvm_context),
                                     llvm::Type::getInt64Ty(_llvm_context),
                                     llvm::Type::getInt64Ty(_llvm_context),
                                     llvm::Type::getInt64Ty(_llvm_context),
                                     llvm::Type::getInt64Ty(_llvm_context)},
                                    "ray_query");
}

std::pair<llvm::Value *, const Type *> HIPCodegenLLVMImpl::_lower_access_chain_address(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_ptr, const Type *type, luisa::span<const xir::Use *const> index_uses) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::hip
