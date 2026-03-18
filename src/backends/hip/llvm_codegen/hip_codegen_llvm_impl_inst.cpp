//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_create_llvm_vector(IB &b, llvm::ArrayRef<llvm::Value *> elems) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_instruction(IB &b, FunctionContext &func_ctx, const xir::Instruction *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::hip
