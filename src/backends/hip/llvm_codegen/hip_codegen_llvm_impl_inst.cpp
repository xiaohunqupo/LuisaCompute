//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_create_llvm_vector(IB &b, llvm::ArrayRef<llvm::Value *> elems) noexcept {
    LUISA_DEBUG_ASSERT(!elems.empty());
    LUISA_DEBUG_ASSERT(std::all_of(elems.begin(), elems.end(), [&](auto e) { return e->getType() == elems.front()->getType(); }));
    auto llvm_vec_type = llvm::FixedVectorType::get(elems.front()->getType(), static_cast<unsigned>(elems.size()));
    auto vec = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_vec_type));
    for (auto i = 0u; i < elems.size(); i++) { vec = b.CreateInsertElement(vec, elems[i], i); }
    return vec;
}

void HIPCodegenLLVMImpl::_translate_instruction(IB &b, FunctionContext &func_ctx, const xir::Instruction *inst) noexcept {
    llvm::Value *result{nullptr};
    auto handle_case = [&]<typename F>(F f) noexcept -> llvm::Value * {
        if constexpr (std::is_same_v<decltype(f()), void>) {
            f();
            return nullptr;
        } else {
            return f();
        }
    };
#define LUISA_HIP_LLVM_TRANSLATE_INST_CASE(inst_type, inst_method)       \
    case xir::inst_type::static_derived_instruction_tag(): {             \
        result = handle_case([&] {                                       \
            return _translate_##inst_method##_inst(                      \
                b, func_ctx, static_cast<const xir::inst_type *>(inst)); \
        });                                                              \
        break;                                                           \
    }
    switch (inst->derived_instruction_tag()) {
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(IfInst, if)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(SwitchInst, switch)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(LoopInst, loop)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(SimpleLoopInst, simple_loop)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(BranchInst, branch)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(ConditionalBranchInst, conditional_branch)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(UnreachableInst, unreachable)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(BreakInst, break)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(ContinueInst, continue)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(ReturnInst, return)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(PhiInst, phi)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(AllocaInst, alloca)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(LoadInst, load)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(StoreInst, store)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(GEPInst, gep)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(AtomicInst, atomic)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(ArithmeticInst, arithmetic)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(ThreadGroupInst, thread_group)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(ResourceQueryInst, resource_query)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(ResourceReadInst, resource_read)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(ResourceWriteInst, resource_write)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(CallInst, call)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(CastInst, cast)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(PrintInst, print)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(ClockInst, clock)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(DebugBreakInst, debug_break)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(AssertInst, assert)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(AssumeInst, assume)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(OutlineInst, outline)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(RayQueryLoopInst, ray_query_loop)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(RayQueryDispatchInst, ray_query_dispatch)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(RayQueryObjectReadInst, ray_query_object_read)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(RayQueryObjectWriteInst, ray_query_object_write)
        LUISA_HIP_LLVM_TRANSLATE_INST_CASE(RayQueryPipelineInst, ray_query_pipeline)
        default: LUISA_ERROR("Unknown instruction tag {}.", xir::to_string(inst->derived_instruction_tag()));
    }
#undef LUISA_HIP_LLVM_TRANSLATE_INST_CASE
    if (result != nullptr) {
        func_ctx.local_values.try_emplace(inst, result);
    }
}

}// namespace luisa::compute::hip
