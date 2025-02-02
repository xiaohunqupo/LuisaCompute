#include <luisa/core/logging.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/builder.h>

#include "helpers.h"

namespace luisa::compute::xir {

AllocaInst *trace_pointer_base_local_alloca_inst(Value *pointer) noexcept {
    if (pointer == nullptr || !pointer->isa<Instruction>()) { return nullptr; }
    switch (auto inst = static_cast<Instruction *>(pointer); inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ALLOCA: {
            if (auto alloca_inst = static_cast<AllocaInst *>(inst); alloca_inst->space() == AllocSpace::LOCAL) {
                return alloca_inst;
            }
            return nullptr;
        }
        case DerivedInstructionTag::GEP: {
            auto gep_inst = static_cast<GEPInst *>(inst);
            return trace_pointer_base_local_alloca_inst(gep_inst->base());
        }
        default: break;
    }
    return nullptr;
}

Instruction *duplicate_instruction(Builder &b, const Instruction *inst,
                                   InstructionCloneValueResolver &resolver) noexcept {
    auto cloned = inst->clone(resolver);
    LUISA_DEBUG_ASSERT(cloned != nullptr, "Failed to clone instruction.");
    b.append(cloned);
    return cloned;
}

bool remove_redundant_phi_instruction(PhiInst *phi) noexcept {
    if (phi->use_list().empty()) {
        phi->remove_self();
        return true;
    }
    static constexpr auto is_invariant = [](Value *v) noexcept {
        if (v == nullptr) { return true; }
        switch (v->derived_value_tag()) {
            case DerivedValueTag::UNDEFINED: [[fallthrough]];
            case DerivedValueTag::CONSTANT: [[fallthrough]];
            case DerivedValueTag::ARGUMENT: [[fallthrough]];
            case DerivedValueTag::SPECIAL_REGISTER: return true;
            default: break;
        }
        return false;
    };
    auto all_same = true;
    auto any_undef = false;
    auto same_incoming = static_cast<Value *>(nullptr);
    for (auto value_use : phi->incoming_value_uses()) {
        auto value = value_use->value();
        LUISA_DEBUG_ASSERT(value != nullptr, "Invalid incoming value.");
        if (same_incoming == nullptr || same_incoming->isa<Undefined>()) { same_incoming = value; }
        if (value->isa<Undefined>()) {
            any_undef = true;
        } else if (same_incoming != value) {
            all_same = false;
            break;
        }
    }
    if (all_same && (!any_undef || is_invariant(same_incoming))) {
        if (same_incoming != nullptr) {
            phi->replace_all_uses_with(same_incoming);
        } else {
            LUISA_DEBUG_ASSERT(phi->use_list().empty(), "Invalid phi node.");
        }
        phi->remove_self();
        return true;
    }
    return false;
}

}// namespace luisa::compute::xir
