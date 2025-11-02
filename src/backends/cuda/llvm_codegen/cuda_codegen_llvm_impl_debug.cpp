//
// Created by mike on 11/1/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

void CUDACodegenLLVMImpl::_translate_print_inst(IB &b, FunctionContext &func_ctx, const xir::PrintInst *inst) noexcept {
    LUISA_WARNING_WITH_LOCATION("Not implemented.");
}

llvm::Value *CUDACodegenLLVMImpl::_translate_clock_inst(IB &b, FunctionContext &func_ctx, const xir::ClockInst *inst) noexcept {
    auto llvm_i64_type = b.getInt64Ty();
    auto llvm_clock = b.CreateIntrinsic(llvm_i64_type, llvm::Intrinsic::nvvm_read_ptx_sreg_clock, {});
    auto llvm_clock_type = _get_llvm_type(inst->type())->reg_type;
    return llvm_clock->getType() == llvm_clock_type ?
               llvm_clock :
               b.CreateZExtOrTrunc(llvm_clock, llvm_clock_type);
}

void CUDACodegenLLVMImpl::_translate_debug_break_inst(IB &b, FunctionContext &func_ctx, const xir::DebugBreakInst *inst) noexcept {
    b.CreateIntrinsic(b.getVoidTy(), llvm::Intrinsic::debugtrap, {});
}

void CUDACodegenLLVMImpl::_translate_assert_inst(IB &b, FunctionContext &func_ctx, const xir::AssertInst *inst) noexcept {
    auto llvm_cond = _get_llvm_value(b, func_ctx, inst->condition());
    auto llvm_msg = llvm::ConstantDataArray::getString(_llvm_context, "Assertion failed: " + inst->message() + "\n");
    // ReSharper disable once CppDFAMemoryLeak
    auto llvm_msg_gv = new llvm::GlobalVariable(
        *_llvm_module, llvm_msg->getType(), true,
        llvm::GlobalValue::PrivateLinkage, llvm_msg, "luisa.assert.message",
        nullptr, llvm::GlobalValue::NotThreadLocal, nvptx_address_space_constant);
    auto llvm_assert_f = _get_assert_function();
    b.CreateCall(llvm_assert_f, {llvm_cond, llvm_msg_gv});
}

void CUDACodegenLLVMImpl::_translate_assume_inst(IB &b, FunctionContext &func_ctx, const xir::AssumeInst *inst) noexcept {
    auto cond = _get_llvm_value(b, func_ctx, inst->condition());
    b.CreateAssumption(cond);
}

}// namespace luisa::compute::cuda
