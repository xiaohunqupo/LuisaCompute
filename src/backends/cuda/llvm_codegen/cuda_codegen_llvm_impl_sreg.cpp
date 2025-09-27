//
// Created by mike on 9/27/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_read_special_register(IB &b, const FunctionContext &func_ctx,
                                                         xir::DerivedSpecialRegisterTag tag) noexcept {
    if (tag == xir::DerivedSpecialRegisterTag::DISPATCH_SIZE) { return func_ctx.llvm_dispatch_size; }
    if (tag == xir::DerivedSpecialRegisterTag::KERNEL_ID) { return func_ctx.llvm_kernel_id; }
    if (tag == xir::DerivedSpecialRegisterTag::WARP_SIZE) { return b.getInt32(32); }
    if (tag == xir::DerivedSpecialRegisterTag::DISPATCH_ID) {
        auto llvm_block_id = _read_special_register(b, func_ctx, xir::DerivedSpecialRegisterTag::BLOCK_ID);
        auto llvm_thread_id = _read_special_register(b, func_ctx, xir::DerivedSpecialRegisterTag::THREAD_ID);
        auto llvm_block_size = _read_special_register(b, func_ctx, xir::DerivedSpecialRegisterTag::BLOCK_SIZE);
        return b.CreateAdd(llvm_thread_id,
                           b.CreateMul(llvm_block_id, llvm_block_size, "", true, true),
                           "", true, true);
    }
    // read from special registers with nvvm intrinsics
    auto [nvvm_sreg_name, nvvm_sreg_type] = [&b, tag]() noexcept -> std::pair<llvm::StringRef, llvm::Type *> {
        auto get_llvm_i32x3_type = [&b]() noexcept {
            return llvm::VectorType::get(b.getInt32Ty(), 3, false);
        };
        switch (tag) {
            case xir::DerivedSpecialRegisterTag::THREAD_ID: return std::make_pair("llvm.nvvm.read.ptx.sreg.tid.", get_llvm_i32x3_type());
            case xir::DerivedSpecialRegisterTag::BLOCK_ID: return std::make_pair("llvm.nvvm.read.ptx.sreg.ctaid.", get_llvm_i32x3_type());
            case xir::DerivedSpecialRegisterTag::WARP_LANE_ID: return std::make_pair("llvm.nvvm.read.ptx.sreg.laneid", b.getInt32Ty());
            case xir::DerivedSpecialRegisterTag::BLOCK_SIZE: return std::make_pair("llvm.nvvm.read.ptx.sreg.ntid.", get_llvm_i32x3_type());
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Unsupported special register.");
    }();
    auto call_nvvm_sreg_func = [this, &b](llvm::StringRef name) noexcept {
        auto nvvm_func = _llvm_module->getFunction(name);
        if (nvvm_func == nullptr) {
            auto nvvm_func_type = llvm::FunctionType::get(b.getInt32Ty(), {}, false);
            nvvm_func = llvm::Function::Create(nvvm_func_type, llvm::Function::ExternalLinkage, 0, name, _llvm_module.get());
            _mark_llvm_function_as_pure(nvvm_func);
        }
        return b.CreateCall(nvvm_func);
    };
    // for vector special registers, we need to call the intrinsic for each axis
    if (nvvm_sreg_type->isVectorTy()) {
        auto llvm_vec = static_cast<llvm::Value *>(llvm::PoisonValue::get(nvvm_sreg_type));
        for (auto [i, axis] : {
                 std::make_pair(0, "x"),
                 std::make_pair(1, "y"),
                 std::make_pair(2, "z"),
             }) {
            auto nvvm_sreg_axis = call_nvvm_sreg_func(nvvm_sreg_name.str().append(axis));
            llvm_vec = b.CreateInsertElement(llvm_vec, nvvm_sreg_axis, i);
        }
        return llvm_vec;
    }
    // scalar special registers are straightforward
    return call_nvvm_sreg_func(nvvm_sreg_name);
}

}// namespace luisa::compute::cuda
