#pragma once

#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>

namespace luisa::compute::xir {

enum struct DerivedFunctionTag {
    KERNEL,
    CALLABLE,
    EXTERNAL,
};

class Module;
class FunctionDefinition;

class LC_XIR_API Function : public IntrusiveForwardNode<Function, DerivedValue<DerivedValueTag::FUNCTION>> {

private:
    Module *_module;
    luisa::vector<Argument *> _arguments;

public:
    explicit Function(Module *module, const Type *type = nullptr) noexcept;
    [[nodiscard]] virtual DerivedFunctionTag derived_function_tag() const noexcept = 0;

    [[nodiscard]] Module *module() noexcept { return _module; }
    [[nodiscard]] const Module *module() const noexcept { return _module; }

    void add_argument(Argument *argument) noexcept;
    void insert_argument(size_t index, Argument *argument) noexcept;
    void remove_argument(Argument *argument) noexcept;
    void remove_argument(size_t index) noexcept;
    void replace_argument(Argument *old_argument, Argument *new_argument) noexcept;
    void replace_argument(size_t index, Argument *argument) noexcept;

    Argument *create_argument(const Type *type, bool by_ref) noexcept;
    ValueArgument *create_value_argument(const Type *type) noexcept;
    ReferenceArgument *create_reference_argument(const Type *type) noexcept;
    ResourceArgument *create_resource_argument(const Type *type) noexcept;

    [[nodiscard]] auto is_definition() const noexcept {
        return derived_function_tag() != DerivedFunctionTag::EXTERNAL;
    }

    [[nodiscard]] auto &arguments() noexcept { return _arguments; }
    [[nodiscard]] auto &arguments() const noexcept { return _arguments; }

    [[nodiscard]] virtual FunctionDefinition *definition() noexcept { return nullptr; }
    [[nodiscard]] const FunctionDefinition *definition() const noexcept {
        return const_cast<Function *>(this)->definition();
    }
};

using FunctionList = IntrusiveForwardList<Function>;

template<DerivedFunctionTag tag, typename Base = Function>
class DerivedFunction : public Base {
public:
    using Base::Base;
    [[nodiscard]] static constexpr DerivedFunctionTag static_derived_function_tag() noexcept { return tag; }
    [[nodiscard]] DerivedFunctionTag derived_function_tag() const noexcept final { return static_derived_function_tag(); }
};

enum struct BasicBlockTraversalOrder {
    PRE_ORDER,
    POST_ORDER,
    REVERSE_PRE_ORDER,
    REVERSE_POST_ORDER,

    // default order
    DEFAULT_ORDER = PRE_ORDER,
};

class LC_XIR_API FunctionDefinition : public Function {

private:
    BasicBlock *_body_block{nullptr};

public:
    using Function::Function;

    void set_body_block(BasicBlock *block) noexcept;
    BasicBlock *create_body_block(bool overwrite_existing = false) noexcept;

    [[nodiscard]] BasicBlock *body_block() noexcept { return _body_block; }
    [[nodiscard]] const BasicBlock *body_block() const noexcept { return _body_block; }

    [[nodiscard]] FunctionDefinition *definition() noexcept final { return this; }

private:
    static void _traverse_basic_block_pre_order(BasicBlock *block, void *visit_ctx,
                                                void (*visit)(void *, BasicBlock *)) noexcept;
    static void _traverse_basic_block_post_order(BasicBlock *block, void *visit_ctx,
                                                 void (*visit)(void *, BasicBlock *)) noexcept;
    static void _traverse_basic_block_reverse_pre_order(BasicBlock *block, void *visit_ctx,
                                                        void (*visit)(void *, BasicBlock *)) noexcept;
    static void _traverse_basic_block_reverse_post_order(BasicBlock *block, void *visit_ctx,
                                                         void (*visit)(void *, BasicBlock *)) noexcept;

public:
    template<typename Visit>
    void traverse_basic_blocks(BasicBlockTraversalOrder order, Visit &&visit) noexcept {
        auto visitor = [](void *ctx, BasicBlock *block) noexcept {
            (*static_cast<Visit *>(ctx))(block);
        };
        switch (order) {
            default: /* pre-order by default */ [[fallthrough]];
            case BasicBlockTraversalOrder::PRE_ORDER:
                _traverse_basic_block_pre_order(_body_block, &visit, visitor);
                break;
            case BasicBlockTraversalOrder::POST_ORDER:
                _traverse_basic_block_post_order(_body_block, &visit, visitor);
                break;
            case BasicBlockTraversalOrder::REVERSE_PRE_ORDER:
                _traverse_basic_block_reverse_pre_order(_body_block, &visit, visitor);
                break;
            case BasicBlockTraversalOrder::REVERSE_POST_ORDER:
                _traverse_basic_block_reverse_post_order(_body_block, &visit, visitor);
                break;
        }
    }
    template<typename Visit>
    void traverse_basic_blocks(BasicBlockTraversalOrder order, Visit &&visit) const noexcept {
        const_cast<FunctionDefinition *>(this)->traverse_basic_blocks(
            order, [&](const BasicBlock *block) noexcept {
                visit(block);
            });
    }
    template<typename Visit>
    void traverse_instructions(BasicBlockTraversalOrder order, Visit &&visit) noexcept {
        traverse_basic_blocks(order, [&visit](BasicBlock *block) noexcept {
            block->traverse_instructions(visit);
        });
    }
    template<typename Visit>
    void traverse_instructions(BasicBlockTraversalOrder order, Visit &&visit) const noexcept {
        traverse_basic_blocks(order, [&visit](const BasicBlock *block) noexcept {
            block->traverse_instructions(visit);
        });
    }

    // traversals with default order
    template<typename Visit>
    void traverse_basic_blocks(Visit &&visit) noexcept {
        traverse_basic_blocks(BasicBlockTraversalOrder::DEFAULT_ORDER, std::forward<Visit>(visit));
    }
    template<typename Visit>
    void traverse_basic_blocks(Visit &&visit) const noexcept {
        traverse_basic_blocks(BasicBlockTraversalOrder::DEFAULT_ORDER, std::forward<Visit>(visit));
    }
    template<typename Visit>
    void traverse_instructions(Visit &&visit) noexcept {
        traverse_instructions(BasicBlockTraversalOrder::DEFAULT_ORDER, std::forward<Visit>(visit));
    }
    template<typename Visit>
    void traverse_instructions(Visit &&visit) const noexcept {
        traverse_instructions(BasicBlockTraversalOrder::DEFAULT_ORDER, std::forward<Visit>(visit));
    }
};

class LC_XIR_API CallableFunction final : public DerivedFunction<DerivedFunctionTag::CALLABLE, FunctionDefinition> {
public:
    using DerivedFunction::DerivedFunction;
};

class LC_XIR_API KernelFunction final : public DerivedFunction<DerivedFunctionTag::KERNEL, FunctionDefinition> {

public:
    static constexpr auto default_block_size = luisa::make_uint3(64u, 1u, 1u);

private:
    std::array<uint, 3> _block_size;

public:
    explicit KernelFunction(Module *module, luisa::uint3 block_size = default_block_size) noexcept;
    void set_block_size(luisa::uint3 size) noexcept;
    [[nodiscard]] luisa::uint3 block_size() const noexcept;
};

class LC_XIR_API ExternalFunction final : public DerivedFunction<DerivedFunctionTag::EXTERNAL> {
public:
    using DerivedFunction::DerivedFunction;
};

}// namespace luisa::compute::xir
