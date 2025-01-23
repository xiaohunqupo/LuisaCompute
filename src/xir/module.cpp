#include <luisa/core/logging.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

Module::Module() noexcept = default;

KernelFunction *Module::create_kernel() noexcept {
    auto f = Pool::current()->create<KernelFunction>(this);
    add_function(f);
    return f;
}

CallableFunction *Module::create_callable(const Type *ret_type) noexcept {
    auto f = Pool::current()->create<CallableFunction>(this, ret_type);
    add_function(f);
    return f;
}

ExternalFunction *Module::create_external_function(const Type *ret_type) noexcept {
    auto f = Pool::current()->create<ExternalFunction>(this, ret_type);
    add_function(f);
    return f;
}

void Module::add_constant(Constant *constant) noexcept {
    LUISA_DEBUG_ASSERT(!constant->is_linked(), "Constant is already linked.");
    _constants.insert_front(constant);
}

Constant *Module::create_constant(const Type *type, const void *data) noexcept {
    auto c = Pool::current()->create<Constant>(type, data);
    add_constant(c);
    return c;
}

Constant *Module::create_constant_zero(const Type *type) noexcept {
    auto c = create_constant(type);
    c->set_zero();
    return c;
}

Constant *Module::create_constant_one(const Type *type) noexcept {
    auto c = create_constant(type);
    c->set_one();
    return c;
}

void Module::add_function(Function *function) noexcept {
    LUISA_DEBUG_ASSERT(!function->is_linked(), "Function is already linked.");
    LUISA_DEBUG_ASSERT(function->module() == this, "Function belongs to another module.");
    _functions.insert_front(function);
}

}// namespace luisa::compute::xir
