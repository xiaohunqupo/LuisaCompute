#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/memory.h>

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace luisa::compute::xir {

class Value;
class Instruction;
class Function;
class Module;
class Constant;

class LC_XIR_API XIRPrinter {

private:
    struct Impl;
    luisa::unique_ptr<Impl> _impl;

public:
    XIRPrinter() noexcept;

    struct NestedBlockFormat {
        bool expand{false};
        int indent{0};
    };

public:
    void reset() noexcept;
    [[nodiscard]] size_t value_uid(const Value *value) noexcept;
    void emit_type(luisa::string &s, const Type *type) noexcept;
    void emit_value_name(luisa::string &s, const Value *value) noexcept;
    void emit_operand(luisa::string &s, const Value *value, NestedBlockFormat block_format = {}) noexcept;
    void emit_instruction(luisa::string &s, const Instruction *instruction, NestedBlockFormat block_format = {}) noexcept;
    void emit_basic_block(luisa::string &s, const BasicBlock *block, NestedBlockFormat block_format = {}) noexcept;
    void emit_constant(luisa::string &s, const Constant *value) noexcept;
    void emit_function_decl(luisa::string &s, const Function *function) noexcept;
    void emit_function(luisa::string &s, const Function *function) noexcept;
    void emit_module(luisa::string &s, const Module *module) noexcept;
};

}// namespace luisa::compute::xir
