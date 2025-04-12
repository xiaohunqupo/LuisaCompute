#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API DebugBreakInst : public DerivedInstruction<DebugBreakInst, DerivedInstructionTag::DEBUG_BREAK> {

public:
    using Evaluate = const void * /* pointer to evaluated data */
        (*)(void * /* backend context */, const char * /* ident */) noexcept;
    using Trap = void (*)() noexcept;
    using Callback = void (*)(void * /* backend context */, Evaluate, Trap);

    struct Watch {
        Value *value;
        luisa::string_view identifier;
    };

    struct ConstWatch {
        const Value *value;
        luisa::string_view identifier;
    };

private:
    Callback _callback;
    luisa::vector<luisa::string> _identifiers;

public:
    explicit DebugBreakInst(BasicBlock *parent_block, Callback callback = nullptr) noexcept;
    void set_callback(Callback callback) noexcept { _callback = callback; }
    void add_watch(luisa::string identifier, xir::Value *value) noexcept;
    void set_watches(luisa::span<const luisa::string> identifiers, luisa::span<xir::Value *const> values) noexcept;
    void reserve_watches(size_t n) noexcept;
    [[nodiscard]] auto callback() const noexcept { return _callback; }
    [[nodiscard]] auto identifiers() const noexcept { return luisa::span{_identifiers}; }
    [[nodiscard]] Watch watch(size_t i) noexcept;
    [[nodiscard]] ConstWatch watch(size_t i) const noexcept;
    [[nodiscard]] auto watch_count() const noexcept { return operand_count(); }
    [[nodiscard]] luisa::string intrinsic_identifier() const noexcept override;
    [[nodiscard]] DebugBreakInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
