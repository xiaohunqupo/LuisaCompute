#include <luisa/core/logging.h>
#include <luisa/xir/use.h>
#include <luisa/xir/module.h>
#include <luisa/xir/value.h>

namespace luisa::compute::xir {

Value::Value(const Type *type) noexcept : _type{type} {}

void Value::replace_all_uses_with(Value *value) noexcept {
    if (value != this) {
        while (!_use_list.empty()) {
            auto use = _use_list.front();
            LUISA_DEBUG_ASSERT(use->value() == this, "Invalid use.");
            auto owned_use = use->remove_self();
            owned_use->set_value(value);
            if (value) { value->use_list().push_front(std::move(owned_use)); }
        }
    }
}

GlobalValueModuleMixin::GlobalValueModuleMixin(Module *module) noexcept : _parent_module{module} {
    LUISA_DEBUG_ASSERT(_parent_module != nullptr, "Module must not be null.");
}

LocalValueFunctionMixin::LocalValueFunctionMixin(Function *function) noexcept : _parent_function{function} {
    LUISA_DEBUG_ASSERT(_parent_function != nullptr, "Function must not be null.");
}

void LocalValueFunctionMixin::_set_parent_function(Function *function) noexcept {
    LUISA_DEBUG_ASSERT(function != nullptr, "Function must not be null.");
    _parent_function = function;
}

Module *LocalValueFunctionMixin::parent_module() noexcept {
    return parent_function()->parent_module();
}

const Module *LocalValueFunctionMixin::parent_module() const noexcept {
    return parent_function()->parent_module();
}

LocalValueBlockMixin::LocalValueBlockMixin(BasicBlock *block) noexcept : _parent_block{block} {
    LUISA_DEBUG_ASSERT(_parent_block != nullptr, "Block must not be null.");
}

void LocalValueBlockMixin::_set_parent_block(BasicBlock *block) noexcept {
    LUISA_DEBUG_ASSERT(block != nullptr, "Block must not be null.");
    _parent_block = block;
}

Function *LocalValueBlockMixin::parent_function() noexcept {
    return parent_block()->parent_function();
}

const Function *LocalValueBlockMixin::parent_function() const noexcept {
    return parent_block()->parent_function();
}

Module *LocalValueBlockMixin::parent_module() noexcept {
    return parent_block()->parent_module();
}

const Module *LocalValueBlockMixin::parent_module() const noexcept {
    return parent_block()->parent_module();
}

}// namespace luisa::compute::xir
