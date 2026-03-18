//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

void HIPCodegenLLVMImpl::_translate_if_inst(IB &b, const FunctionContext &func_ctx, const xir::IfInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_switch_inst(IB &b, const FunctionContext &func_ctx, const xir::SwitchInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_loop_inst(IB &b, const FunctionContext &func_ctx, const xir::LoopInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_simple_loop_inst(IB &b, const FunctionContext &func_ctx, const xir::SimpleLoopInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_branch_inst(IB &b, const FunctionContext &func_ctx, const xir::BranchInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_conditional_branch_inst(IB &b, const FunctionContext &func_ctx, const xir::ConditionalBranchInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_unreachable_inst(IB &b, FunctionContext &func_ctx, const xir::UnreachableInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_break_inst(IB &b, const FunctionContext &func_ctx, const xir::BreakInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_continue_inst(IB &b, const FunctionContext &func_ctx, const xir::ContinueInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_return_inst(IB &b, const FunctionContext &func_ctx, const xir::ReturnInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::hip
