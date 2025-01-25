#pragma once

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class AllocaInst;
class Value;
class Instruction;
class Builder;

struct InstructionCloneValueResolver;

[[nodiscard]] LC_XIR_API AllocaInst *trace_pointer_base_local_alloca_inst(Value *pointer) noexcept;

[[nodiscard]] LC_XIR_API Instruction *duplicate_instruction(Builder &b, const Instruction *inst,
                                                            InstructionCloneValueResolver &resolver) noexcept;

}// namespace luisa::compute::xir
