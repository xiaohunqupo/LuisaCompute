//
// Created by mike on 11/1/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

void CUDACodegenLLVMImpl::_translate_autodiff_scope_inst(IB &b, FunctionContext &func_ctx, const xir::AutodiffScopeInst *inst) noexcept {
    LUISA_ERROR_WITH_LOCATION("Autodiff scope instruction should have been lowered.");
}

llvm::Value *CUDACodegenLLVMImpl::_translate_autodiff_intrinsic_inst(IB &b, FunctionContext &func_ctx, const xir::AutodiffIntrinsicInst *inst) noexcept {
    LUISA_ERROR_WITH_LOCATION("Autodiff intrinsic instruction should have been lowered.");
}

}// namespace luisa::compute::cuda
