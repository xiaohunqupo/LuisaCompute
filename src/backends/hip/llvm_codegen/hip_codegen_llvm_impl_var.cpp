//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_translate_alloca_inst(IB &b, FunctionContext &func_ctx, const xir::AllocaInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_load_inst(IB &b, const FunctionContext &func_ctx, const xir::LoadInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_store_inst(IB &b, const FunctionContext &func_ctx, const xir::StoreInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_gep_inst(IB &b, FunctionContext &func_ctx, const xir::GEPInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_load_llvm_value(IB &b, llvm::Value *llvm_ptr, const Type *type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_store_llvm_value(IB &b, llvm::Value *llvm_ptr, llvm::Value *llvm_value, const Type *type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_create_temp_in_alloca_block(const FunctionContext &func_ctx, llvm::Type *t, size_t align) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::hip
