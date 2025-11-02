//
// Created by mike on 11/1/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_translate_alloca_inst(IB &b, FunctionContext &func_ctx, const xir::AllocaInst *inst) noexcept {
    auto llvm_type = _get_llvm_type(inst->type())->mem_type;
    if (inst->is_local()) {
        auto llvm_alloca = b.CreateAlloca(llvm_type, nullptr, inst->name().value_or(""));
        llvm_alloca->setAlignment(llvm::Align{inst->type()->alignment()});
        return llvm_alloca;
    }
    // shared alloca's are mapped to global variables in address space nvvm_address_space_shared
    auto llvm_global = new llvm::GlobalVariable{
        *_llvm_module, llvm_type, false, llvm::GlobalValue::PrivateLinkage,
        nullptr, inst->name().value_or("shared"), nullptr,
        llvm::GlobalValue::NotThreadLocal, nvptx_address_space_shared, false};
    llvm_global->setAlignment(llvm::Align{inst->type()->alignment()});
    llvm_global->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
    return llvm_global;
}

llvm::Value *CUDACodegenLLVMImpl::_translate_load_inst(IB &b, FunctionContext &func_ctx, const xir::LoadInst *inst) noexcept {
    auto llvm_type = _get_llvm_type(inst->type());
    auto llvm_ptr = func_ctx.get_local_value<llvm::Value>(inst->variable());
    auto llvm_mem_v = b.CreateAlignedLoad(llvm_type->mem_type, llvm_ptr,
                                          llvm::Align{inst->type()->alignment()},
                                          inst->name().value_or(""));
    return _convert_llvm_mem_value_to_reg(b, llvm_mem_v, llvm_type->reg_type);
}

void CUDACodegenLLVMImpl::_translate_store_inst(IB &b, FunctionContext &func_ctx, const xir::StoreInst *inst) noexcept {
    auto type = inst->value()->type();
    auto llvm_type = _get_llvm_type(type);
    auto llvm_ptr = func_ctx.get_local_value<llvm::Value>(inst->variable());
    auto llvm_reg_v = func_ctx.get_local_value<llvm::Value>(inst->value());
    auto llvm_mem_v = _convert_llvm_reg_value_to_mem(b, llvm_reg_v, llvm_type->mem_type);
    b.CreateAlignedStore(llvm_mem_v, llvm_ptr, llvm::Align{type->alignment()});
}

llvm::Value *CUDACodegenLLVMImpl::_translate_gep_inst(IB &b, FunctionContext &func_ctx, const xir::GEPInst *inst) noexcept {
    // FIXME: we directly calculate the address here, which might hinder LLVM optimizations
    auto base = inst->base();
    auto llvm_ptr = func_ctx.get_local_value<llvm::Value>(base);
    LUISA_DEBUG_ASSERT(llvm_ptr->getType()->isPointerTy());
    auto type = base->type();
    for (auto index_use : inst->index_uses()) {
        auto llvm_index = _get_llvm_value(b, func_ctx, index_use->value());
        switch (type->tag()) {
            case Type::Tag::VECTOR: {
                type = type->element();
                auto llvm_elem_type = _get_llvm_type(type)->mem_type;
                llvm_ptr = b.CreateInBoundsGEP(llvm_elem_type, llvm_ptr, {llvm_index});
                break;
            }
            case Type::Tag::MATRIX: {
                type = Type::vector(type->element(), type->dimension());
                auto llvm_col_type = _get_llvm_type(type)->mem_type;
                llvm_ptr = b.CreateInBoundsGEP(llvm_col_type, llvm_ptr, {llvm_index});
                break;
            }
            case Type::Tag::ARRAY: {
                type = type->element();
                auto llvm_elem_type = _get_llvm_type(type)->mem_type;
                llvm_ptr = b.CreateInBoundsGEP(llvm_elem_type, llvm_ptr, {llvm_index});
                break;
            }
            case Type::Tag::STRUCTURE: {
                LUISA_DEBUG_ASSERT(llvm::isa<llvm::ConstantInt>(llvm_index));
                auto member_index = llvm::cast<llvm::ConstantInt>(llvm_index)->getZExtValue();
                LUISA_DEBUG_ASSERT(member_index < type->members().size());
                auto llvm_struct_info = _get_llvm_type(type);
                auto llvm_member_offset = llvm_struct_info->member_offsets[member_index];
                type = type->members()[member_index];
                auto llvm_i8_type = b.getInt8Ty();
                llvm_ptr = b.CreateConstInBoundsGEP1_64(llvm_i8_type, llvm_ptr, llvm_member_offset);
                break;
            }
            default: LUISA_ERROR("Invalid GEP base type: {}.", type->description());
        }
    }
    return llvm_ptr;
}

}// namespace luisa::compute::cuda
