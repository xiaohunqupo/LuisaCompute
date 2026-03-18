//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

size_t HIPCodegenLLVMImpl::_get_type_alignment(const Type *type) noexcept {
    return _data_layout->getABITypeAlign(_get_llvm_type(type)->mem_type).value();
}

const HIPCodegenLLVMImpl::LLVMTypeInfo *HIPCodegenLLVMImpl::_get_llvm_type(const Type *type) noexcept {
    if (auto iter = _xir_to_llvm_type.find(type); iter != _xir_to_llvm_type.end()) {
        return iter->second.get();
    }
    auto llvm_type_info = [this, type]() noexcept -> luisa::unique_ptr<LLVMTypeInfo> {
        auto llvm_type = [this, type]() noexcept -> llvm::Type * {
            switch (type->tag()) {
                case Type::Tag::BOOL: return llvm::Type::getInt1Ty(_llvm_context);
                case Type::Tag::INT8: return llvm::Type::getInt8Ty(_llvm_context);
                case Type::Tag::INT16: return llvm::Type::getInt16Ty(_llvm_context);
                case Type::Tag::INT32: return llvm::Type::getInt32Ty(_llvm_context);
                case Type::Tag::INT64: return llvm::Type::getInt64Ty(_llvm_context);
                case Type::Tag::UINT8: return llvm::Type::getInt8Ty(_llvm_context);
                case Type::Tag::UINT16: return llvm::Type::getInt16Ty(_llvm_context);
                case Type::Tag::UINT32: return llvm::Type::getInt32Ty(_llvm_context);
                case Type::Tag::UINT64: return llvm::Type::getInt64Ty(_llvm_context);
                case Type::Tag::FLOAT16: return llvm::Type::getHalfTy(_llvm_context);
                case Type::Tag::FLOAT32: return llvm::Type::getFloatTy(_llvm_context);
                case Type::Tag::FLOAT64: return llvm::Type::getDoubleTy(_llvm_context);
                case Type::Tag::VECTOR: {
                    auto v = static_cast<const VectorType *>(type);
                    auto elem_type = _get_llvm_type(v->element())->mem_type;
                    return llvm::VectorType::get(elem_type, v->dimension());
                }
                case Type::Tag::MATRIX: {
                    auto m = static_cast<const MatrixType *>(type);
                    auto elem_type = _get_llvm_type(m->element())->mem_type;
                    auto rows = m->rows();
                    llvm::SmallVector<llvm::Type *, 4> fields;
                    for (size_t i = 0; i < rows; i++) {
                        fields.push_back(llvm::ArrayType::get(elem_type, rows));
                    }
                    return llvm::StructType::create(fields);
                }
                case Type::Tag::ARRAY: {
                    auto a = static_cast<const ArrayType *>(type);
                    auto elem_type = _get_llvm_type(a->element())->mem_type;
                    return llvm::ArrayType::get(elem_type, a->dimension());
                }
                case Type::Tag::STRUCT: {
                    auto s = static_cast<const StructType *>(type);
                    llvm::SmallVector<llvm::Type *, 4> fields;
                    llvm::SmallVector<size_t, 4> offsets;
                    for (auto member : s->members()) {
                        fields.push_back(_get_llvm_type(member)->mem_type);
                    }
                    auto struct_type = llvm::StructType::create(fields, s->name().value_or(""));
                    return struct_type;
                }
                default: LUISA_ERROR_WITH_LOCATION("Unsupported type.");
            }
        }();
        auto info = luisa::make_unique<LLVMTypeInfo>();
        info->mem_type = llvm_type;
        info->reg_type = llvm_type;
        return info;
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
    return llvm::StructType::create({llvm::ArrayType::get(llvm::VectorType::get(llvm::Type::getFloatTy(_llvm_context), 4), 3),
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
                                     llvm::VectorType::get(llvm::Type::getFloatTy(_llvm_context), 2),
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
                                     llvm::VectorType::get(llvm::Type::getFloatTy(_llvm_context), 2),
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
