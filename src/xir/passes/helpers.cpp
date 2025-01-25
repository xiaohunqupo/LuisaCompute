#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/gep.h>
#include "helpers.h"

namespace luisa::compute::xir {

AllocaInst *trace_pointer_base_local_alloca_inst(Value *pointer) noexcept {
    if (pointer == nullptr || pointer->derived_value_tag() != DerivedValueTag::INSTRUCTION) {
        return nullptr;
    }
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
                                   InstructionDuplicatorValueResolver &resolver) noexcept {
    return nullptr;
}

}// namespace luisa::compute::xir
