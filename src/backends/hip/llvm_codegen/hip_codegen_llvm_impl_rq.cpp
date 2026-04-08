//
// Created by mike on 4/8/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

void HIPCodegenLLVMImpl::_translate_ray_query_loop_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryLoopInst *inst) noexcept {
    b.GetInsertBlock()->setName("ray.query.loop");
    auto llvm_dispatch_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->dispatch_block());
    llvm_dispatch_block->setName("ray.query.dispatch");
    b.CreateBr(llvm_dispatch_block);
}

void HIPCodegenLLVMImpl::_translate_ray_query_dispatch_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryDispatchInst *inst) noexcept {
    // luisa.ray.query.proceed();
    // switch (luisa.ray.query.state()) {
    //    case surface: br surface_block
    //    case procedural: br procedural_block
    //    default: br exit_block
    // }
    _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_proceed, b.getVoidTy(), {});
    auto llvm_state = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_state, b.getInt8Ty(), {});
    auto llvm_exit_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->exit_block());
    llvm_exit_block->setName("ray.query.exit");
    auto llvm_surface_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->on_surface_candidate_block());
    llvm_surface_block->setName("ray.query.on.surface.candidate");
    auto llvm_procedural_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->on_procedural_candidate_block());
    llvm_procedural_block->setName("ray.query.on.procedural.candidate");
    auto llvm_dispatch = b.CreateSwitch(llvm_state, llvm_exit_block, 2);
    llvm_dispatch->addCase(b.getInt8(llvm_ray_query_state_surface_candidate), llvm_surface_block);
    llvm_dispatch->addCase(b.getInt8(llvm_ray_query_state_procedural_candidate), llvm_procedural_block);
}

llvm::Value *HIPCodegenLLVMImpl::_translate_ray_query_object_read_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectReadInst *inst) noexcept {
    LUISA_DEBUG_ASSERT(inst->operand_count() == 1);
    auto op = inst->op();

    switch (op) {
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE:
            return _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_is_surface_candidate, b.getInt1Ty(), {});
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE:
            return _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_is_procedural_candidate, b.getInt1Ty(), {});
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED:
            return _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_is_terminated, b.getInt1Ty(), {});
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY: {
            auto ox = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_ray_origin_x, b.getFloatTy(), {});
            auto oy = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_ray_origin_y, b.getFloatTy(), {});
            auto oz = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_ray_origin_z, b.getFloatTy(), {});
            auto tmin = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_ray_tmin, b.getFloatTy(), {});
            auto dx = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_ray_direction_x, b.getFloatTy(), {});
            auto dy = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_ray_direction_y, b.getFloatTy(), {});
            auto dz = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_ray_direction_z, b.getFloatTy(), {});
            auto tmax = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_ray_tmax, b.getFloatTy(), {});
            auto llvm_f32x3_array_type = llvm::ArrayType::get(b.getFloatTy(), 3);
            auto origin = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_f32x3_array_type));
            origin = b.CreateInsertValue(origin, ox, 0);
            origin = b.CreateInsertValue(origin, oy, 1);
            origin = b.CreateInsertValue(origin, oz, 2);
            auto direction = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_f32x3_array_type));
            direction = b.CreateInsertValue(direction, dx, 0);
            direction = b.CreateInsertValue(direction, dy, 1);
            direction = b.CreateInsertValue(direction, dz, 2);
            auto result_type = _get_llvm_ray_type();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            result = b.CreateInsertValue(result, origin, llvm_ray_type_origin_index);
            result = b.CreateInsertValue(result, tmin, llvm_ray_type_t_min_index);
            result = b.CreateInsertValue(result, direction, llvm_ray_type_direction_index);
            result = b.CreateInsertValue(result, tmax, llvm_ray_type_t_max_index);
            return result;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT: {
            auto inst_id = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_candidate_inst_id, b.getInt32Ty(), {});
            auto prim_id = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_candidate_prim_id, b.getInt32Ty(), {});
            auto u = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_candidate_bary_u, b.getFloatTy(), {});
            auto v = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_candidate_bary_v, b.getFloatTy(), {});
            auto t = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_candidate_hit_t, b.getFloatTy(), {});
            auto bary = _create_llvm_vector(b, {u, v});
            auto result_type = _get_llvm_surface_hit_type();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            result = b.CreateInsertValue(result, inst_id, llvm_surface_hit_type_inst_id_index);
            result = b.CreateInsertValue(result, prim_id, llvm_surface_hit_type_prim_id_index);
            result = b.CreateInsertValue(result, bary, llvm_surface_hit_type_bary_index);
            result = b.CreateInsertValue(result, t, llvm_surface_hit_type_t_index);
            return result;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT: {
            auto inst_id = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_candidate_inst_id, b.getInt32Ty(), {});
            auto prim_id = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_candidate_prim_id, b.getInt32Ty(), {});
            auto result_type = _get_llvm_procedural_hit_type();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            result = b.CreateInsertValue(result, inst_id, llvm_procedural_hit_type_inst_id_index);
            result = b.CreateInsertValue(result, prim_id, llvm_procedural_hit_type_prim_id_index);
            return result;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT: {
            auto inst_id = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_committed_inst_id, b.getInt32Ty(), {});
            auto prim_id = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_committed_prim_id, b.getInt32Ty(), {});
            auto hit_kind = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_committed_hit_kind, b.getInt32Ty(), {});
            // Read committed hit float fields through wrapper function calls,
            // then pass each result through an inline asm barrier to make
            // the value opaque to the optimizer.  Without this barrier, the
            // LLVM O2 pipeline (specifically FunctionAttrs + downstream DCE)
            // eliminates the entire barycentric-interpolation → hit-position
            // → shadow-ray computation chain because the float values only
            // feed into FP math (no observable memory side-effects), unlike
            // the integer fields which are used for buffer GEP+load.
            auto u_raw = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_committed_bary_u, b.getFloatTy(), {});
            auto u = _create_opaque_float_barrier(b, u_raw, "committed.bary.u");
            auto v_raw = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_committed_bary_v, b.getFloatTy(), {});
            auto v = _create_opaque_float_barrier(b, v_raw, "committed.bary.v");
            auto t_raw = _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_committed_hit_t, b.getFloatTy(), {});
            auto t = _create_opaque_float_barrier(b, t_raw, "committed.hit.t");
            auto bary = _create_llvm_vector(b, {u, v});
            auto result_type = _get_llvm_committed_hit_type();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            result = b.CreateInsertValue(result, inst_id, llvm_committed_hit_type_inst_id_index);
            result = b.CreateInsertValue(result, prim_id, llvm_committed_hit_type_prim_id_index);
            result = b.CreateInsertValue(result, bary, llvm_committed_hit_type_bary_index);
            result = b.CreateInsertValue(result, hit_kind, llvm_committed_hit_type_hit_kind_index);
            result = b.CreateInsertValue(result, t, llvm_committed_hit_type_t_index);
            return result;
        }
        default: break;
    }
    LUISA_ERROR("Invalid op (code = {}) for RayQueryObjectReadInst.", luisa::to_underlying(op));
}

void HIPCodegenLLVMImpl::_translate_ray_query_object_write_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectWriteInst *inst) noexcept {
    auto intrinsic = [op = inst->op()] {
        switch (op) {
            case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE: return llvm_ray_query_intrinsic_name_commit_surface_hit;
            case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL: return llvm_ray_query_intrinsic_name_commit_procedural_hit;
            case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE: return llvm_ray_query_intrinsic_name_terminate;
            case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED: return llvm_ray_query_intrinsic_name_proceed;
            default: break;
        }
        LUISA_ERROR("Invalid op (code = {}) for RayQueryObjectWriteInst.", luisa::to_underlying(op));
    }();
    LUISA_DEBUG_ASSERT(inst->type() == nullptr);
    LUISA_DEBUG_ASSERT(inst->operand_count() == 1 || inst->operand_count() == 2);
    llvm::SmallVector<llvm::Value *, 2> llvm_args;
    for (auto &&op_use : inst->operand_uses().subspan(1) /* skip the query object */) {
        llvm_args.emplace_back(_get_llvm_value(b, func_ctx, op_use->value()));
    }
    _call_ray_query_intrinsic(b, func_ctx, intrinsic, b.getVoidTy(), llvm_args);
}

void HIPCodegenLLVMImpl::_translate_ray_query_pipeline_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryPipelineInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_call_ray_query_intrinsic(IB &b, FunctionContext &func_ctx, llvm::StringRef name, llvm::Type *ret, llvm::ArrayRef<llvm::Value *> args) noexcept {
    auto generic_ptr_type = b.getPtrTy(0);
    auto llvm_state_ptr = b.CreateAddrSpaceCast(func_ctx.llvm_rq_state, generic_ptr_type, "rq.state.generic");
    llvm::SmallVector<llvm::Value *, 8> augmented_args;
    augmented_args.push_back(llvm_state_ptr);
    augmented_args.append(args.begin(), args.end());
    auto func = _llvm_module->getFunction(name);
    if (func == nullptr) {
        llvm::SmallVector<llvm::Type *, 8> arg_types;
        for (auto arg : augmented_args) { arg_types.push_back(arg->getType()); }
        auto func_type = llvm::FunctionType::get(ret, arg_types, false);
        func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, name, _llvm_module.get());
    }
    return b.CreateCall(func, augmented_args);
}

llvm::Value *HIPCodegenLLVMImpl::_create_opaque_float_barrier(IB &b, llvm::Value *val, const llvm::Twine &name) noexcept {
    auto *float_ty = b.getFloatTy();
    auto *asm_func_ty = llvm::FunctionType::get(float_ty, {float_ty}, false);
    auto *ia = llvm::InlineAsm::get(asm_func_ty, "v_mov_b32 $0, $1", "=v,v", /*hasSideEffects=*/true);
    return b.CreateCall(asm_func_ty, ia, {val}, name);
}

}// namespace luisa::compute::hip
