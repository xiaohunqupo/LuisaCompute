//
// Created by mike on 9/27/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_create_llvm_vector(IB &b, llvm::ArrayRef<llvm::Value *> elems) noexcept {
    LUISA_DEBUG_ASSERT(elems.size() >= 2 && elems.size() <= 4);
    LUISA_DEBUG_ASSERT(std::all_of(elems.begin(), elems.end(), [&](auto e) { return e->getType() == elems.front()->getType(); }));
    auto llvm_vec_type = llvm::VectorType::get(elems.front()->getType(), static_cast<unsigned>(elems.size()), false);
    auto vec = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_vec_type));
    for (auto i = 0u; i < elems.size(); i++) { vec = b.CreateInsertElement(vec, elems[i], i); }
    return vec;
}

llvm::Value *CUDACodegenLLVMImpl::_convert_llvm_reg_value_to_mem(IB &b, llvm::Value *reg_v, llvm::Type *mem_type) noexcept {
    auto reg_type = reg_v->getType();
    if (reg_type == mem_type) { return reg_v; }
    // i1 scalars and vectors are stored as i8 in memory
    if (reg_type->isIntOrIntVectorTy(1)) {
        LUISA_DEBUG_ASSERT(mem_type->isIntOrIntVectorTy(8));
        return b.CreateZExt(reg_v, mem_type, reg_v->getName().str() + ".to.mem");
    }
    // otherwise it must be i64/f64 vector types
    LUISA_DEBUG_ASSERT(reg_type->isVectorTy());
    auto elem_type = llvm::cast<llvm::VectorType>(reg_type)->getElementType();
    auto elem_count = llvm::cast<llvm::VectorType>(reg_type)->getElementCount().getFixedValue();
    LUISA_DEBUG_ASSERT(elem_count == 3 || elem_count == 4);
    // i64/f64 vectors are stored as padded arrays in memory
    auto padded_elem_count = elem_count == 3 ? 4 : elem_count;
    LUISA_DEBUG_ASSERT(mem_type->isArrayTy() &&
                       mem_type->getArrayElementType() == elem_type &&
                       mem_type->getArrayNumElements() == padded_elem_count);
    auto mem_v = static_cast<llvm::Value *>(llvm::PoisonValue::get(mem_type));
    for (auto i = 0u; i < elem_count; i++) {
        auto reg_elem = b.CreateExtractElement(reg_v, i);
        mem_v = b.CreateInsertValue(mem_v, reg_elem, i);
    }
    mem_v->setName(reg_v->getName().str() + ".to.mem");
    return mem_v;
}

llvm::Value *CUDACodegenLLVMImpl::_convert_llvm_mem_value_to_reg(IB &b, llvm::Value *mem_v, llvm::Type *reg_type) noexcept {
    auto mem_type = mem_v->getType();
    if (mem_type == reg_type) { return mem_v; }
    // i1 scalars and vectors are stored as i8 in memory
    if (reg_type->isIntOrIntVectorTy(1)) {
        LUISA_DEBUG_ASSERT(mem_type->isIntOrIntVectorTy(8));
        return b.CreateTrunc(mem_v, reg_type, mem_v->getName().str() + ".to.reg");
    }
    // otherwise it must be i64/f64 vector types
    LUISA_DEBUG_ASSERT(reg_type->isVectorTy());
    auto elem_type = llvm::cast<llvm::VectorType>(reg_type)->getElementType();
    auto elem_count = llvm::cast<llvm::VectorType>(reg_type)->getElementCount().getFixedValue();
    LUISA_DEBUG_ASSERT(elem_count == 3 || elem_count == 4);
    // i64/f64 vectors are stored as padded arrays in memory
    auto padded_elem_count = elem_count == 3 ? 4 : elem_count;
    LUISA_DEBUG_ASSERT(mem_type->isArrayTy() &&
                       mem_type->getArrayElementType() == elem_type &&
                       mem_type->getArrayNumElements() == padded_elem_count);
    auto reg_v = static_cast<llvm::Value *>(llvm::PoisonValue::get(reg_type));
    for (auto i = 0u; i < elem_count; i++) {
        auto mem_elem = b.CreateExtractValue(mem_v, i);
        reg_v = b.CreateInsertElement(reg_v, mem_elem, i);
    }
    reg_v->setName(mem_v->getName().str() + ".to.reg");
    return reg_v;
}

void CUDACodegenLLVMImpl::_translate_instruction(IB &b, FunctionContext &func_ctx, const xir::Instruction *inst) noexcept {
    llvm::Value *result{nullptr};
    auto handle_case = [&]<typename F>(F f) noexcept -> llvm::Value * {
        if constexpr (std::is_invocable_r_v<void, F>) {
            f();
            return nullptr;
        } else {
            return f();
        }
    };
#define LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(inst_type, inst_method)      \
    case xir::inst_type::static_derived_instruction_tag(): {             \
        result = handle_case([&] {                                       \
            return _translate_##inst_method##_inst(                      \
                b, func_ctx, static_cast<const xir::inst_type *>(inst)); \
        });                                                              \
        break;                                                           \
    }
    switch (inst->derived_instruction_tag()) {
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(IfInst, if)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(SwitchInst, switch)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(LoopInst, loop)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(SimpleLoopInst, simple_loop)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(BranchInst, branch)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ConditionalBranchInst, conditional_branch)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(UnreachableInst, unreachable)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(BreakInst, break)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ContinueInst, continue)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ReturnInst, return)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(RasterDiscardInst, raster_discard)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(PhiInst, phi)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(AllocaInst, alloca)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(LoadInst, load)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(StoreInst, store)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(GEPInst, gep)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(AtomicInst, atomic)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ArithmeticInst, arithmetic)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ThreadGroupInst, thread_group)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ResourceQueryInst, resource_query)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ResourceReadInst, resource_read)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ResourceWriteInst, resource_write)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(RayQueryLoopInst, ray_query_loop)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(RayQueryDispatchInst, ray_query_dispatch)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(RayQueryObjectReadInst, ray_query_object_read)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(RayQueryObjectWriteInst, ray_query_object_write)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(RayQueryPipelineInst, ray_query_pipeline)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(AutodiffScopeInst, autodiff_scope)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(AutodiffIntrinsicInst, autodiff_intrinsic)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(CallInst, call)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(CastInst, cast)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(PrintInst, print)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ClockInst, clock)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(DebugBreakInst, debug_break)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(AssertInst, assert)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(AssumeInst, assume)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(OutlineInst, outline)
        default: LUISA_ERROR("Unknown instruction tag {}.", xir::to_string(inst->derived_instruction_tag()));
    }
#undef LUISA_CUDA_LLVM_TRANSLATE_INST_CASE
    if (result != nullptr) {
        func_ctx.local_values.try_emplace(inst, result);
    }
}

void CUDACodegenLLVMImpl::_translate_if_inst(IB &b, FunctionContext &func_ctx, const xir::IfInst *inst) noexcept {
    auto llvm_cond = func_ctx.get_local_value<llvm::Value>(inst->condition());
    auto llvm_true_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->true_block());
    auto llvm_false_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->false_block());
    b.CreateCondBr(llvm_cond, llvm_true_block, llvm_false_block);
}

void CUDACodegenLLVMImpl::_translate_switch_inst(IB &b, FunctionContext &func_ctx, const xir::SwitchInst *inst) noexcept {
}

void CUDACodegenLLVMImpl::_translate_loop_inst(IB &b, FunctionContext &func_ctx, const xir::LoopInst *inst) noexcept {
}

void CUDACodegenLLVMImpl::_translate_simple_loop_inst(IB &b, FunctionContext &func_ctx, const xir::SimpleLoopInst *inst) noexcept {
}

void CUDACodegenLLVMImpl::_translate_branch_inst(IB &b, FunctionContext &func_ctx, const xir::BranchInst *inst) noexcept {
    auto llvm_target = func_ctx.get_local_value<llvm::BasicBlock>(inst->target_block());
    b.CreateBr(llvm_target);
}

void CUDACodegenLLVMImpl::_translate_conditional_branch_inst(IB &b, FunctionContext &func_ctx, const xir::ConditionalBranchInst *inst) noexcept {
    auto llvm_cond = func_ctx.get_local_value<llvm::Value>(inst->condition());
    auto llvm_true_target = func_ctx.get_local_value<llvm::BasicBlock>(inst->true_block());
    auto llvm_false_target = func_ctx.get_local_value<llvm::BasicBlock>(inst->false_block());
    b.CreateCondBr(llvm_cond, llvm_true_target, llvm_false_target);
}

void CUDACodegenLLVMImpl::_translate_unreachable_inst(IB &b, FunctionContext &func_ctx, const xir::UnreachableInst *inst) noexcept {
    b.CreateUnreachable();
}

void CUDACodegenLLVMImpl::_translate_break_inst(IB &b, FunctionContext &func_ctx, const xir::BreakInst *inst) noexcept {
    auto llvm_target = func_ctx.get_local_value<llvm::BasicBlock>(inst->target_block());
    b.CreateBr(llvm_target);
}

void CUDACodegenLLVMImpl::_translate_continue_inst(IB &b, FunctionContext &func_ctx, const xir::ContinueInst *inst) noexcept {
    auto llvm_target = func_ctx.get_local_value<llvm::BasicBlock>(inst->target_block());
    b.CreateBr(llvm_target);
}

void CUDACodegenLLVMImpl::_translate_return_inst(IB &b, FunctionContext &func_ctx, const xir::ReturnInst *inst) noexcept {
    if (auto v = inst->return_value()) {
        auto llvm_ret_v = func_ctx.get_local_value<llvm::Value>(v);
        b.CreateRet(llvm_ret_v);
    } else {
        b.CreateRetVoid();
    }
}

llvm::PHINode *CUDACodegenLLVMImpl::_translate_phi_inst(IB &b, FunctionContext &func_ctx, const xir::PhiInst *inst) noexcept {
    func_ctx.pending_phi_nodes.emplace_back(inst);
    auto llvm_type = _get_llvm_type(inst->type());
    return b.CreatePHI(llvm_type->reg_type, inst->incoming_count(), inst->name().value_or(""));
}

void CUDACodegenLLVMImpl::_finalize_pending_phi_nodes(const FunctionContext &func_ctx) noexcept {
    for (auto phi : func_ctx.pending_phi_nodes) {
        auto llvm_phi = func_ctx.get_local_value<llvm::PHINode>(phi);
        for (auto i = 0u; i < phi->incoming_count(); i++) {
            auto [value, block] = phi->incoming(i);
            auto llvm_value = func_ctx.get_local_value<llvm::Value>(value);
            auto llvm_block = func_ctx.get_local_value<llvm::BasicBlock>(block);
            llvm_phi->addIncoming(llvm_value, llvm_block);
        }
    }
}

llvm::Value *CUDACodegenLLVMImpl::_translate_alloca_inst(IB &b, FunctionContext &func_ctx, const xir::AllocaInst *inst) noexcept {
    auto llvm_type = _get_llvm_type(inst->type())->mem_type;
    auto llvm_alloca = b.CreateAlloca(llvm_type, nullptr, inst->name().value_or(""));
    llvm_alloca->setAlignment(llvm::Align{inst->type()->alignment()});
    return llvm_alloca;
}

llvm::Value *CUDACodegenLLVMImpl::_translate_load_inst(IB &b, FunctionContext &func_ctx, const xir::LoadInst *inst) noexcept {
    auto llvm_type = _get_llvm_type(inst->type());
    auto llvm_ptr = func_ctx.get_local_value<llvm::Value>(inst->variable());
    auto llvm_mem_v = b.CreateAlignedLoad(llvm_type->mem_type, llvm_ptr,
                                          llvm::Align{inst->type()->alignment()},
                                          inst->name().value_or(""));
    return _convert_llvm_mem_value_to_reg(b, llvm_mem_v, llvm_type->reg_type);
}

void CUDACodegenLLVMImpl::_translate_store_inst(IB &b, FunctionContext &func_ctx, const xir::StoreInst *inst) noexcept {
    auto type = inst->value()->type();
    auto llvm_type = _get_llvm_type(type);
    auto llvm_ptr = func_ctx.get_local_value<llvm::Value>(inst->variable());
    auto llvm_reg_v = func_ctx.get_local_value<llvm::Value>(inst->value());
    auto llvm_mem_v = _convert_llvm_reg_value_to_mem(b, llvm_reg_v, llvm_type->mem_type);
    b.CreateAlignedStore(llvm_mem_v, llvm_ptr, llvm::Align{type->alignment()});
}

llvm::Value *CUDACodegenLLVMImpl::_translate_gep_inst(IB &b, FunctionContext &func_ctx, const xir::GEPInst *inst) noexcept {
}

llvm::Value *CUDACodegenLLVMImpl::_translate_atomic_inst(IB &b, FunctionContext &func_ctx, const xir::AtomicInst *inst) noexcept {
}

llvm::Value *CUDACodegenLLVMImpl::_translate_arithmetic_inst(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept {
}

llvm::Value *CUDACodegenLLVMImpl::_translate_thread_group_inst(IB &b, FunctionContext &func_ctx, const xir::ThreadGroupInst *inst) noexcept {
}

llvm::Value *CUDACodegenLLVMImpl::_translate_resource_query_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceQueryInst *inst) noexcept {
}

llvm::Value *CUDACodegenLLVMImpl::_translate_resource_read_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceReadInst *inst) noexcept {
}

void CUDACodegenLLVMImpl::_translate_resource_write_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceWriteInst *inst) noexcept {
}

void CUDACodegenLLVMImpl::_translate_ray_query_loop_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryLoopInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void CUDACodegenLLVMImpl::_translate_ray_query_dispatch_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryDispatchInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *CUDACodegenLLVMImpl::_translate_ray_query_object_read_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectReadInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void CUDACodegenLLVMImpl::_translate_ray_query_object_write_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectWriteInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void CUDACodegenLLVMImpl::_translate_ray_query_pipeline_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryPipelineInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *CUDACodegenLLVMImpl::_translate_call_inst(IB &b, FunctionContext &func_ctx, const xir::CallInst *inst) noexcept {
}

llvm::Value *CUDACodegenLLVMImpl::_translate_cast_inst(IB &b, FunctionContext &func_ctx, const xir::CastInst *inst) noexcept {
}

void CUDACodegenLLVMImpl::_translate_print_inst(IB &b, FunctionContext &func_ctx, const xir::PrintInst *inst) noexcept {
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
    auto llvm_cond = func_ctx.get_local_value<llvm::Value>(inst->condition());
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
    auto cond = func_ctx.get_local_value<llvm::Value>(inst->condition());
    b.CreateAssumption(cond);
}

void CUDACodegenLLVMImpl::_translate_raster_discard_inst(IB &b, FunctionContext &func_ctx, const xir::RasterDiscardInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED("Raster discard instruction is not supported in CUDA backend.");
}

void CUDACodegenLLVMImpl::_translate_autodiff_scope_inst(IB &b, FunctionContext &func_ctx, const xir::AutodiffScopeInst *inst) noexcept {
    LUISA_ERROR_WITH_LOCATION("Autodiff scope instruction should have been lowered.");
}

llvm::Value *CUDACodegenLLVMImpl::_translate_autodiff_intrinsic_inst(IB &b, FunctionContext &func_ctx, const xir::AutodiffIntrinsicInst *inst) noexcept {
    LUISA_ERROR_WITH_LOCATION("Autodiff intrinsic instruction should have been lowered.");
}

void CUDACodegenLLVMImpl::_translate_outline_inst(IB &b, FunctionContext &func_ctx, const xir::OutlineInst *inst) noexcept {
    LUISA_ERROR_WITH_LOCATION("Outline instruction should have been lowered.");
}

}// namespace luisa::compute::cuda
