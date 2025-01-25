#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instructions/branch.h>

namespace luisa::compute::xir {

BranchInst *BranchInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    auto cloned = Pool::current()->create<BranchInst>();
    auto resolved_target = resolver.resolve(target_block());
    LUISA_DEBUG_ASSERT(resolved_target == nullptr || resolved_target->derived_value_tag() == DerivedValueTag::BASIC_BLOCK, "Invalid target block.");
    cloned->set_target_block(static_cast<BasicBlock *>(resolved_target));
    return cloned;
}

ConditionalBranchInst *ConditionalBranchInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    auto resolved_cond = resolver.resolve(condition());
    auto cloned = Pool::current()->create<ConditionalBranchInst>(resolved_cond);
    auto resolved_true_block = resolver.resolve(true_block());
    LUISA_DEBUG_ASSERT(resolved_true_block == nullptr || resolved_true_block->derived_value_tag() == DerivedValueTag::BASIC_BLOCK, "Invalid true block");
    auto resolved_false_block = resolver.resolve(false_block());
    LUISA_DEBUG_ASSERT(resolved_false_block == nullptr || resolved_false_block->derived_value_tag() == DerivedValueTag::BASIC_BLOCK, "Invalid false block");
    cloned->set_true_target(static_cast<BasicBlock *>(resolved_true_block));
    cloned->set_false_target(static_cast<BasicBlock *>(resolved_false_block));
    return cloned;
}

}// namespace luisa::compute::xir
