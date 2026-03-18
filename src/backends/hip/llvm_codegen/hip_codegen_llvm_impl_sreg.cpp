//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_read_special_register(IB &b, const FunctionContext &func_ctx,
                                                        xir::DerivedSpecialRegisterTag tag) noexcept {
    switch (tag) {
        case xir::DerivedSpecialRegisterTag::THREAD_ID: return _read_thread_id(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::BLOCK_ID: return _read_block_id(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::WARP_LANE_ID: return _read_warp_lane_id(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::DISPATCH_ID: return _read_dispatch_id(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::KERNEL_ID: return _read_kernel_id(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::BLOCK_SIZE: return _read_block_size(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::WARP_SIZE: return _read_warp_size(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::DISPATCH_SIZE: return _read_dispatch_size(b, func_ctx);
        default: break;
    }
    LUISA_NOT_IMPLEMENTED("Special register {} not implemented.", xir::to_string(tag));
}

llvm::Value *HIPCodegenLLVMImpl::_read_block_id(IB &b, const FunctionContext &) noexcept {
    auto llvm_i32_type = b.getInt32Ty();
    auto llvm_wgid_x = b.CreateIntrinsic(llvm_i32_type, llvm::Intrinsic::amdgcn_workgroup_id_x, {}, {}, "sreg.block.id.x");
    auto llvm_wgid_y = b.CreateIntrinsic(llvm_i32_type, llvm::Intrinsic::amdgcn_workgroup_id_y, {}, {}, "sreg.block.id.y");
    auto llvm_wgid_z = b.CreateIntrinsic(llvm_i32_type, llvm::Intrinsic::amdgcn_workgroup_id_z, {}, {}, "sreg.block.id.z");
    llvm::SmallVector<llvm::Value *, 3> elems;
    elems.push_back(llvm_wgid_x);
    elems.push_back(llvm_wgid_y);
    elems.push_back(llvm_wgid_z);
    auto llvm_bid = _create_llvm_vector(b, elems);
    llvm_bid->setName("sreg.block.id");
    return llvm_bid;
}

llvm::Value *HIPCodegenLLVMImpl::_read_block_size(IB &, const FunctionContext &) noexcept {
    auto [bx, by, bz] = _config.block_size;
    return llvm::ConstantDataVector::get(_llvm_context, std::array{bx, by, bz});
}

llvm::Value *HIPCodegenLLVMImpl::_read_thread_id(IB &b, const FunctionContext &) noexcept {
    auto llvm_i32_type = b.getInt32Ty();
    auto llvm_wi_x = b.CreateIntrinsic(llvm_i32_type, llvm::Intrinsic::amdgcn_workitem_id_x, {}, {}, "sreg.thread.id.x");
    auto llvm_wi_y = b.CreateIntrinsic(llvm_i32_type, llvm::Intrinsic::amdgcn_workitem_id_y, {}, {}, "sreg.thread.id.y");
    auto llvm_wi_z = b.CreateIntrinsic(llvm_i32_type, llvm::Intrinsic::amdgcn_workitem_id_z, {}, {}, "sreg.thread.id.z");
    llvm::SmallVector<llvm::Value *, 3> elems;
    elems.push_back(llvm_wi_x);
    elems.push_back(llvm_wi_y);
    elems.push_back(llvm_wi_z);
    auto llvm_tid = _create_llvm_vector(b, elems);
    llvm_tid->setName("sreg.thread.id");
    return llvm_tid;
}

llvm::Value *HIPCodegenLLVMImpl::_read_dispatch_size(IB &, const FunctionContext &func_ctx) noexcept {
    return func_ctx.llvm_dispatch_size;
}

llvm::Value *HIPCodegenLLVMImpl::_read_dispatch_id(IB &b, const FunctionContext &func_ctx) noexcept {
    auto llvm_tid = _read_thread_id(b, func_ctx);
    auto llvm_bid = _read_block_id(b, func_ctx);
    auto llvm_block_size = _read_block_size(b, func_ctx);
    auto llvm_dispatch_offset = b.CreateMul(llvm_bid, llvm_block_size, "", true, true);
    auto llvm_dispatch_id = b.CreateAdd(llvm_tid, llvm_dispatch_offset, "sreg.dispatch.id", true, true);
    return llvm_dispatch_id;
}

llvm::Value *HIPCodegenLLVMImpl::_read_warp_size(IB &b, const FunctionContext &) const noexcept {
    return b.getInt32(32);
}

llvm::Value *HIPCodegenLLVMImpl::_read_warp_lane_id(IB &b, const FunctionContext &) const noexcept {
    auto llvm_i32_type = b.getInt32Ty();
    auto llvm_result = b.CreateIntrinsic(llvm_i32_type, llvm::Intrinsic::amdgcn_workitem_id_x, {}, {}, "sreg.warp.lane.id");
    return llvm_result;
}

llvm::Value *HIPCodegenLLVMImpl::_read_kernel_id(IB &, const FunctionContext &func_ctx) noexcept {
    return func_ctx.llvm_kernel_id;
}

}// namespace luisa::compute::hip
