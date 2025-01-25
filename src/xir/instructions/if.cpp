#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instructions/if.h>

namespace luisa::compute::xir {

IfInst *IfInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    auto resolved_cond = resolver.resolve(condition());
    auto cloned = Pool::current()->create<IfInst>(resolved_cond);
    auto resolved_true_block = resolver.resolve(true_block());
    LUISA_DEBUG_ASSERT(resolved_true_block == nullptr || resolved_true_block->derived_value_tag() == DerivedValueTag::BASIC_BLOCK, "Invalid true block.");
    auto resolved_false_block = resolver.resolve(false_block());
    LUISA_DEBUG_ASSERT(resolved_false_block == nullptr || resolved_false_block->derived_value_tag() == DerivedValueTag::BASIC_BLOCK, "Invalid false block.");
    auto resolved_merge_block = resolver.resolve(merge_block());
    LUISA_DEBUG_ASSERT(resolved_merge_block == nullptr || resolved_merge_block->derived_value_tag() == DerivedValueTag::BASIC_BLOCK, "Invalid merge block.");
    cloned->set_true_target(static_cast<BasicBlock *>(resolved_true_block));
    cloned->set_false_target(static_cast<BasicBlock *>(resolved_false_block));
    cloned->set_merge_block(static_cast<BasicBlock *>(resolved_merge_block));
    return cloned;
}

}// namespace luisa::compute::xir
