#pragma once

#include <luisa/core/managed_ilist.h>
#include <luisa/xir/traits.h>

namespace luisa::compute::xir {

class Value;
class User;

class LUISA_XIR_API Use final : public ManagedIntrusiveForwardNode<Use> {

private:
    User *_user;
    Value *_value;

public:
    explicit Use(User *user, Value *value = nullptr) noexcept;
    void set_value(Value *value) noexcept;

    [[nodiscard]] auto value() noexcept {
        validate_canary();
        return _value;
    }
    [[nodiscard]] auto value() const noexcept {
        return const_cast<const Value *>(_value);
    }
    [[nodiscard]] auto user() noexcept {
        validate_canary();
        return _user;
    }
    [[nodiscard]] auto user() const noexcept {
        return const_cast<const User *>(_user);
    }
};

using UseList = ManagedIntrusiveForwardList<Use>;

}// namespace luisa::compute::xir
