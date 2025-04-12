#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/debug_break.h>

namespace luisa::compute::xir {

DebugBreakInst::DebugBreakInst(BasicBlock *parent_block, Callback callback) noexcept
    : DerivedInstruction{parent_block, nullptr}, _callback{callback} {}

void DebugBreakInst::add_watch(luisa::string identifier, xir::Value *value) noexcept {
    add_operand(value);
    _identifiers.emplace_back(std::move(identifier));
}

void DebugBreakInst::set_watches(luisa::span<const luisa::string> identifiers, luisa::span<xir::Value *const> values) noexcept {
    LUISA_DEBUG_ASSERT(identifiers.size() == values.size(), "Invalid size.");
    set_operands(values);
    _identifiers.resize(identifiers.size());
    std::copy(identifiers.begin(), identifiers.end(), _identifiers.begin());
}

void DebugBreakInst::reserve_watches(size_t n) noexcept {
    reserve_operands(n);
    _identifiers.reserve(n);
}

DebugBreakInst::Watch DebugBreakInst::watch(size_t i) noexcept {
    LUISA_DEBUG_ASSERT(i < _identifiers.size(), "Invalid index.");
    return {operand(i), _identifiers[i]};
}

DebugBreakInst::ConstWatch DebugBreakInst::watch(size_t i) const noexcept {
    LUISA_DEBUG_ASSERT(i < _identifiers.size(), "Invalid index.");
    return {operand(i), _identifiers[i]};
}

luisa::string DebugBreakInst::intrinsic_identifier() const noexcept {
    LUISA_DEBUG_ASSERT(_identifiers.size() == operand_count(), "Invalid size.");
    auto ident = Super::intrinsic_identifier();
    if (!_identifiers.empty()) {
        ident.append("(");
        for (auto &&i : _identifiers) {
            luisa::format_to(std::back_inserter(ident), "{:?}, ", i);
        }
        ident.pop_back();
        ident.pop_back();
        ident.append(")");
    }
    return ident;
}

DebugBreakInst *DebugBreakInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    auto cloned = b.debug_break(_callback);
    cloned->reserve_watches(operand_count());
    for (auto i = 0u; i < operand_count(); i++) {
        auto cloned_op = resolver.resolve(operand(i));
        cloned->add_watch(_identifiers[i], cloned_op);
    }
    return cloned;
}

}// namespace luisa::compute::xir
