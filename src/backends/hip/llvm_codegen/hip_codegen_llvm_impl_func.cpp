//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Function *HIPCodegenLLVMImpl::_get_or_declare_llvm_function(const xir::Function *func) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Function *HIPCodegenLLVMImpl::_declare_llvm_kernel_function(const xir::KernelFunction *func) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Function *HIPCodegenLLVMImpl::_declare_llvm_callable_function(const xir::CallableFunction *func) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Function *HIPCodegenLLVMImpl::_declare_llvm_external_function(const xir::ExternalFunction *func) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Function *HIPCodegenLLVMImpl::_translate_function(const xir::FunctionDefinition *func) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Function *HIPCodegenLLVMImpl::_translate_kernel_function(const xir::KernelFunction *func) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Function *HIPCodegenLLVMImpl::_translate_callable_function(const xir::CallableFunction *func) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::BasicBlock *HIPCodegenLLVMImpl::_translate_function_definition(FunctionContext &func_ctx, const xir::FunctionDefinition *f) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_mark_llvm_function_as_pure(llvm::Function *func) noexcept {
}

llvm::Function *HIPCodegenLLVMImpl::_get_assert_function() noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Function *HIPCodegenLLVMImpl::_get_vprintf_function() noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture2d_read_function(llvm::VectorType *llvm_value_type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture2d_write_function(llvm::VectorType *llvm_value_type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture3d_read_function(llvm::VectorType *llvm_value_type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture3d_write_function(llvm::VectorType *llvm_value_type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::InlineAsm *HIPCodegenLLVMImpl::_get_inline_asm(std::string_view asm_string, std::string_view constraints, bool has_side_effects) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_call_inst(IB &b, FunctionContext &func_ctx, const xir::CallInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_outline_inst(IB &b, FunctionContext &func_ctx, const xir::OutlineInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::hip
