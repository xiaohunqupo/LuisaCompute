#include <luisa/core/logging.h>
#include <luisa/xir/user.h>

namespace luisa::compute::xir {

void User::set_operand(size_t index, Value *value) noexcept {
    LUISA_DEBUG_ASSERT(index < _operands.size(), "Index out of range.");
    LUISA_DEBUG_ASSERT(_operands[index] != nullptr && _operands[index]->user() == this, "Invalid operand.");
    set_operand_use_value(_operands[index].get(), value);
}

Use *User::operand_use(size_t index) noexcept {
    LUISA_DEBUG_ASSERT(index < _operands.size(), "Index out of range.");
    return _operands[index].get();
}

const Use *User::operand_use(size_t index) const noexcept {
    LUISA_DEBUG_ASSERT(index < _operands.size(), "Index out of range.");
    return _operands[index].get();
}

Value *User::operand(size_t index) noexcept {
    return operand_use(index)->value();
}

const Value *User::operand(size_t index) const noexcept {
    return operand_use(index)->value();
}

void User::set_operand_use_value(Use *use, Value *value) noexcept {
    LUISA_DEBUG_ASSERT(use != nullptr, "Invalid use.");
    if (use->value() != value) {
        auto owned_use = use->value() ? use->remove_self() : use->lock();
        owned_use->set_value(value);
        if (value && owned_use->user()->_should_add_self_to_operand_use_lists()) {
            value->use_list().push_front(std::move(owned_use));
        }
    }
}

void User::set_operand_count(size_t n) noexcept {
    if (n < _operands.size()) {// remove redundant operands
        for (auto i = n; i < _operands.size(); i++) {
            set_operand_use_value(_operands[i].get(), nullptr);
        }
        _operands.resize(n);
    } else {// create new operands
        _operands.reserve(n);
        for (auto i = _operands.size(); i < n; i++) {
            _operands.emplace_back(luisa::make_managed<Use>(this));
        }
    }
}

void User::set_operands(luisa::span<Value *const> operands) noexcept {
    set_operand_count(operands.size());
    for (auto i = 0u; i < operands.size(); i++) {
        set_operand(i, operands[i]);
    }
}

void User::reserve_operands(size_t n) noexcept {
    _operands.reserve(n);
}

void User::add_operand(Value *value) noexcept {
    auto use = luisa::make_managed<Use>(this);
    set_operand_use_value(use.get(), value);
    _operands.emplace_back(std::move(use));
}

void User::insert_operand(size_t index, Value *value) noexcept {
    LUISA_DEBUG_ASSERT(index <= _operands.size(), "Index out of range.");
    auto use = luisa::make_managed<Use>(this);
    set_operand_use_value(use.get(), value);
    _operands.insert(_operands.cbegin() + index, std::move(use));
}

void User::remove_operand(size_t index) noexcept {
    if (index < _operands.size()) {
        set_operand_use_value(_operands[index].get(), nullptr);
        _operands.erase(_operands.cbegin() + index);
    }
}

luisa::span<Use *> User::operand_uses() noexcept {
    return to_unowned_span(luisa::span{_operands});
}

luisa::span<const Use *const> User::operand_uses() const noexcept {
    return to_unowned_span(luisa::span{_operands});
}

size_t User::operand_count() const noexcept {
    return _operands.size();
}

}// namespace luisa::compute::xir
