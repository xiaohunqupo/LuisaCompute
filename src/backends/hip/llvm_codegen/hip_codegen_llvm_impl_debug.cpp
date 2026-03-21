//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

void HIPCodegenLLVMImpl::_translate_print_inst(IB &b, FunctionContext &func_ctx, const xir::PrintInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_clock_inst(IB &b, FunctionContext &func_ctx, const xir::ClockInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_debug_break_inst(IB &b, FunctionContext &func_ctx, const xir::DebugBreakInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_assert_inst(IB &b, FunctionContext &func_ctx, const xir::AssertInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_assume_inst(IB &b, FunctionContext &func_ctx, const xir::AssumeInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_create_assertion_with_message(IB &b, llvm::Value *cond, luisa::string_view message) noexcept {
    (void)b;
    (void)cond;
    (void)message;
}

}// namespace luisa::compute::hip
