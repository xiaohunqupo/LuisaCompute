//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_read_special_register(IB &b, const FunctionContext &func_ctx, xir::DerivedSpecialRegisterTag tag) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_read_block_id(IB &b, const FunctionContext &func_ctx) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_read_block_size(IB &b, const FunctionContext &func_ctx) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_read_thread_id(IB &b, const FunctionContext &func_ctx) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_read_dispatch_size(IB &b, const FunctionContext &func_ctx) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_read_dispatch_id(IB &b, const FunctionContext &func_ctx) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_read_warp_size(IB &b, const FunctionContext &func_ctx) const noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_read_warp_lane_id(IB &b, const FunctionContext &func_ctx) const noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_read_kernel_id(IB &b, const FunctionContext &func_ctx) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_thread_group_inst(IB &b, FunctionContext &func_ctx, const xir::ThreadGroupInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::hip
