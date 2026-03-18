//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_translate_arithmetic_inst(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_outer_product(IB &b, llvm::Value *lhs, llvm::Value *rhs) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_matrix_multiply(IB &b, llvm::Value *lhs, llvm::Value *rhs) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_matrix_determinant(IB &b, llvm::Value *m) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_matrix_transpose(IB &b, llvm::Value *m) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_matrix_inverse(IB &b, llvm::Value *m) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_aggregate(IB &b, const FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_shuffle(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_insert(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_extract(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::hip
