#include "luisa/core/stl/unordered_map.h"

#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/xir/function.h>

namespace luisa::compute::xir {

Function::Function(Module *module, const Type *type) noexcept
    : Super{type}, _module{module} {
    LUISA_DEBUG_ASSERT(module != nullptr, "Module must not be null.");
}

void Function::add_argument(Argument *argument) noexcept {
    argument->_set_parent_function(this);
    _arguments.emplace_back(argument);
}

void Function::insert_argument(size_t index, Argument *argument) noexcept {
    argument->_set_parent_function(this);
    _arguments.insert(_arguments.begin() + index, argument);
}

void Function::remove_argument(Argument *argument) noexcept {
    for (auto i = 0u; i < _arguments.size(); i++) {
        if (_arguments[i] == argument) {
            remove_argument(i);
            return;
        }
    }
    LUISA_ERROR_WITH_LOCATION("Argument not found.");
}

void Function::remove_argument(size_t index) noexcept {
    LUISA_ASSERT(index < _arguments.size(), "Argument index out of range.");
    _arguments[index]->_set_parent_function(nullptr);
    _arguments.erase(_arguments.begin() + index);
}

void Function::replace_argument(Argument *old_argument, Argument *new_argument) noexcept {
    if (old_argument != new_argument) {
        for (auto i = 0u; i < _arguments.size(); i++) {
            if (_arguments[i] == old_argument) {
                replace_argument(i, new_argument);
                return;
            }
        }
        LUISA_ERROR_WITH_LOCATION("Argument not found.");
    }
}

void Function::replace_argument(size_t index, Argument *argument) noexcept {
    LUISA_ASSERT(index < _arguments.size(), "Argument index out of range.");
    _arguments[index]->_set_parent_function(nullptr);
    _arguments[index]->replace_all_uses_with(argument);
    argument->_set_parent_function(this);
    _arguments[index] = argument;
}

Argument *Function::create_argument(const Type *type, bool by_ref) noexcept {
    if (type->is_resource()) {
        LUISA_ASSERT(!by_ref, "Resource argument must not be passed by reference.");
        return create_resource_argument(type);
    }
    return by_ref ? static_cast<Argument *>(create_reference_argument(type)) :
                    static_cast<Argument *>(create_value_argument(type));
}

ValueArgument *Function::create_value_argument(const Type *type) noexcept {
    LUISA_ASSERT(!type->is_resource(), "Resource argument must be created with create_resource_argument.");
    LUISA_ASSERT(!type->is_custom(), "Opaque argument must be created with create_reference_argument.");
    auto argument = Pool::current()->create<ValueArgument>(type, this);
    add_argument(argument);
    return argument;
}

ReferenceArgument *Function::create_reference_argument(const Type *type) noexcept {
    LUISA_ASSERT(!type->is_resource(), "Resource argument must be created with create_resource_argument.");
    auto argument = Pool::current()->create<ReferenceArgument>(type, this);
    add_argument(argument);
    return argument;
}

ResourceArgument *Function::create_resource_argument(const Type *type) noexcept {
    LUISA_ASSERT(type->is_resource(), "Resource argument must be created with create_resource_argument.");
    auto argument = Pool::current()->create<ResourceArgument>(type, this);
    add_argument(argument);
    return argument;
}

void FunctionDefinition::set_body_block(BasicBlock *block) noexcept {
    _body_block = block;
}

BasicBlock *FunctionDefinition::create_body_block(bool overwrite_existing) noexcept {
    LUISA_ASSERT(_body_block == nullptr || overwrite_existing,
                 "Body block already exists.");
    auto new_block = Pool::current()->create<BasicBlock>();
    set_body_block(new_block);
    return new_block;
}

namespace detail {

void traverse_basic_block_pre_order(luisa::unordered_set<BasicBlock *> &visited, BasicBlock *block,
                                    void *visit_ctx, void (*visit)(void *, BasicBlock *)) noexcept {
    if (visited.emplace(block).second) {
        visit(visit_ctx, block);
        auto terminator = block->terminator();
        for (auto use : terminator->operand_uses()) {
            if (auto v = use->value();
                v != nullptr && v->derived_value_tag() == DerivedValueTag::BASIC_BLOCK) {
                traverse_basic_block_pre_order(visited, static_cast<BasicBlock *>(v), visit_ctx, visit);
            }
        }
    }
}

void traverse_basic_block_post_order(luisa::unordered_set<BasicBlock *> &visited, BasicBlock *block,
                                     void *visit_ctx, void (*visit)(void *, BasicBlock *)) noexcept {
    if (visited.emplace(block).second) {
        auto terminator = block->terminator();
        for (auto use : terminator->operand_uses()) {
            if (auto v = use->value();
                v != nullptr && v->derived_value_tag() == DerivedValueTag::BASIC_BLOCK) {
                traverse_basic_block_post_order(visited, static_cast<BasicBlock *>(v), visit_ctx, visit);
            }
        }
        visit(visit_ctx, block);
    }
}

}// namespace detail

void FunctionDefinition::_traverse_basic_block_pre_order(BasicBlock *block, void *visit_ctx,
                                                         void (*visit)(void *, BasicBlock *)) noexcept {
    luisa::unordered_set<BasicBlock *> visited;
    detail::traverse_basic_block_pre_order(visited, block, visit_ctx, visit);
}

void FunctionDefinition::_traverse_basic_block_post_order(BasicBlock *block, void *visit_ctx, void (*visit)(void *, BasicBlock *)) noexcept {
    luisa::unordered_set<BasicBlock *> visited;
    detail::traverse_basic_block_post_order(visited, block, visit_ctx, visit);
}

void FunctionDefinition::_traverse_basic_block_reverse_pre_order(BasicBlock *block, void *visit_ctx, void (*visit)(void *, BasicBlock *)) noexcept {
    luisa::vector<BasicBlock *> stack;
    _traverse_basic_block_pre_order(block, &stack, [](void *ctx, BasicBlock *bb) noexcept {
        static_cast<luisa::vector<BasicBlock *> *>(ctx)->emplace_back(bb);
    });
    for (auto iter = stack.rbegin(); iter != stack.rend(); ++iter) {
        visit(visit_ctx, *iter);
    }
}

void FunctionDefinition::_traverse_basic_block_reverse_post_order(BasicBlock *block, void *visit_ctx, void (*visit)(void *, BasicBlock *)) noexcept {
    luisa::vector<BasicBlock *> stack;
    _traverse_basic_block_post_order(block, &stack, [](void *ctx, BasicBlock *bb) noexcept {
        static_cast<luisa::vector<BasicBlock *> *>(ctx)->emplace_back(bb);
    });
    for (auto iter = stack.rbegin(); iter != stack.rend(); ++iter) {
        visit(visit_ctx, *iter);
    }
}

KernelFunction::KernelFunction(Module *module, luisa::uint3 block_size) noexcept
    : DerivedFunction{module}, _block_size{} { set_block_size(block_size); }

void KernelFunction::set_block_size(luisa::uint3 size) noexcept {
    auto thread_count = size.x * size.y * size.z;
    LUISA_ASSERT(thread_count >= 32u &&
                     thread_count <= 1024u &&
                     thread_count % 32u == 0u,
                 "Invalid block size: {}.", size);
    _block_size = {size.x, size.y, size.z};
}

luisa::uint3 KernelFunction::block_size() const noexcept {
    return luisa::make_uint3(_block_size[0], _block_size[1], _block_size[2]);
}

}// namespace luisa::compute::xir
