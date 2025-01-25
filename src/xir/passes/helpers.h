#pragma once

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class AllocaInst;
class Value;
class Instruction;
class Builder;

[[nodiscard]] LC_XIR_API AllocaInst *trace_pointer_base_local_alloca_inst(Value *pointer) noexcept;

struct InstructionDuplicatorValueResolver {
    virtual ~InstructionDuplicatorValueResolver() noexcept = default;
    [[nodiscard]] virtual Value *resolve(const Value *value) noexcept = 0;
};

[[nodiscard]] LC_XIR_API Instruction *duplicate_instruction(Builder &b, const Instruction *inst,
                                                            InstructionDuplicatorValueResolver &resolver) noexcept;

}// namespace luisa::compute::xir
