//
// Created by mike on 11/1/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_translate_thread_group_inst(IB &b, FunctionContext &func_ctx, const xir::ThreadGroupInst *inst) noexcept {
    switch (inst->op()) {
        case xir::ThreadGroupOp::SHADER_EXECUTION_REORDER: {
            // no-op in CUDA, while asm sideeffect "call (), _optix_hitobject_reorder, ($0,$1);", "r,r"(i32 %3, i32 %4) in ray tracing shaders
            if (_config.enable_ray_tracing) {
                auto llvm_void_type = b.getVoidTy();
                auto llvm_i32_type = b.getInt32Ty();
                auto llvm_asm_type = llvm::FunctionType::get(llvm_void_type, {llvm_i32_type, llvm_i32_type}, false);
                auto llvm_asm = llvm::InlineAsm::get(
                    llvm_asm_type,
                    "call (), _optix_hitobject_reorder, ($0,$1);",
                    "r,r",
                    true);
                auto arg_count = inst->operand_count();
                auto llvm_hint = arg_count >= 1 ?
                                     _get_llvm_value(b, func_ctx, inst->operand(0)) :
                                     llvm::ConstantInt::get(llvm_i32_type, 0);
                auto llvm_hint_bits = arg_count >= 2 ?
                                          _get_llvm_value(b, func_ctx, inst->operand(1)) :
                                          llvm::ConstantInt::get(llvm_i32_type, 0);
                return b.CreateCall(llvm_asm, {llvm_hint, llvm_hint_bits});
            }
            return nullptr;
        }
        case xir::ThreadGroupOp::RASTER_QUAD_DDX: LUISA_NOT_IMPLEMENTED();
        case xir::ThreadGroupOp::RASTER_QUAD_DDY: LUISA_NOT_IMPLEMENTED();
        case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE: {// lane_id == ctz(activemask)
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_first_active_lane_id = b.CreateUnaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask);
            auto llvm_lane_id = _read_warp_lane_id(b, func_ctx);
            return b.CreateICmpEQ(llvm_lane_id, llvm_first_active_lane_id);
        }
        case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE: {// ctz(activemask)
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask);
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL: break;
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: break;
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: break;
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: break;
        case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS: {// popc(ballot(activemask, pred))
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_ballot = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::nvvm_vote_ballot_sync, {llvm_active_mask, llvm_pred});
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::ctpop, llvm_ballot);
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_MAX: break;
        case xir::ThreadGroupOp::WARP_ACTIVE_MIN: break;
        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT: break;
        case xir::ThreadGroupOp::WARP_ACTIVE_SUM: break;
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL: {
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            return b.CreateIntrinsic(b.getInt1Ty(), llvm::Intrinsic::nvvm_vote_all_sync, {llvm_active_mask, llvm_pred});
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY: {
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            return b.CreateIntrinsic(b.getInt1Ty(), llvm::Intrinsic::nvvm_vote_any_sync, {llvm_active_mask, llvm_pred});
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK: {// uint4(ballot(activemask, pred), 0, 0, 0)
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_ballot = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::nvvm_vote_ballot_sync, {llvm_active_mask, llvm_pred});
            auto llvm_zero = llvm::Constant::getNullValue(llvm::VectorType::get(b.getInt32Ty(), 4, false));
            return b.CreateInsertElement(llvm_zero, llvm_ballot, b.getInt64(0));
        }
        case xir::ThreadGroupOp::WARP_PREFIX_COUNT_BITS: {// popc(ballot(activemask, pred) & lane_mask)
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_prefix_mask = _read_warp_prefix_lane_mask(b);
            auto llvm_ballot = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::nvvm_vote_ballot_sync, {llvm_active_mask, llvm_pred});
            auto llvm_ballot_and_prefix = b.CreateAnd(llvm_ballot, llvm_prefix_mask);
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::ctpop, llvm_ballot_and_prefix);
        }
        case xir::ThreadGroupOp::WARP_PREFIX_SUM: break;
        case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT: break;
        case xir::ThreadGroupOp::WARP_READ_LANE: break;
        case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE: break;
        case xir::ThreadGroupOp::SYNCHRONIZE_BLOCK: {
#if LLVM_VERSION_MAJOR >= 21
            return b.CreateIntrinsic(llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all, {b.getInt32(0)});
#else
            return b.CreateIntrinsic(llvm::Intrinsic::nvvm_barrier0, {});
#endif
        }
    }
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::cuda
