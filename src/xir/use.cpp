#include <luisa/core/logging.h>
#include <luisa/xir/value.h>
#include <luisa/xir/user.h>
#include <luisa/xir/use.h>

namespace luisa::compute::xir {

Use::Use(User *user, Value *value) noexcept : _user{user}, _value{value} {
    LUISA_DEBUG_ASSERT(user != nullptr, "User must not be null.");
}

void Use::set_value(Value *value) noexcept {
    validate_canary();
    _value = value;
}

}// namespace luisa::compute::xir
