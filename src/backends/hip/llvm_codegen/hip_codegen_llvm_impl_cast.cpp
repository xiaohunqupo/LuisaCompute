//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_convert_llvm_reg_value_to_mem(IB &b, llvm::Value *reg_v, const Type *type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_convert_llvm_mem_value_to_reg(IB &b, llvm::Value *mem_v, const Type *type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_bitwise_cast(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_static_cast(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_static_cast_scalar_to_scalar(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_static_cast_scalar_to_vector(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_static_cast_vector_to_vector(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_texel_cast(IB &b, llvm::Value *llvm_src, llvm::Type *dst_type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_cast_inst(IB &b, FunctionContext &func_ctx, const xir::CastInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::hip
