#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instructions/outline.h>

namespace luisa::compute::xir {

OutlineInst *OutlineInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    auto cloned = Pool::current()->create<OutlineInst>();
    auto resolved_target = resolver.resolve(target_block());
    LUISA_DEBUG_ASSERT(resolved_target == nullptr || resolved_target->derived_value_tag() == DerivedValueTag::BASIC_BLOCK, "Invalid target block.");
    auto resolved_merge = resolver.resolve(merge_block());
    LUISA_DEBUG_ASSERT(resolved_merge == nullptr || resolved_merge->derived_value_tag() == DerivedValueTag::BASIC_BLOCK, "Invalid merge block.");
    cloned->set_target_block(static_cast<BasicBlock *>(resolved_target));
    cloned->set_merge_block(static_cast<BasicBlock *>(resolved_merge));
    return cloned;
}

}// namespace luisa::compute::xir
