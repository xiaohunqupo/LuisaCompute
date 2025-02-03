#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class Function;

enum struct DerivedArgumentTag {
    VALUE,
    REFERENCE,
    RESOURCE,
};

class LC_XIR_API Argument : public DerivedFunctionScopeValue<Argument, DerivedValueTag::ARGUMENT> {
public:
    Argument(Function *parent_function, const Type *type) noexcept;
    [[nodiscard]] virtual DerivedArgumentTag derived_argument_tag() const noexcept = 0;
    [[nodiscard]] bool is_lvalue() const noexcept final {
        return derived_argument_tag() == DerivedArgumentTag::REFERENCE;
    }
    [[nodiscard]] auto is_value() const noexcept { return derived_argument_tag() == DerivedArgumentTag::VALUE; }
    [[nodiscard]] auto is_reference() const noexcept { return derived_argument_tag() == DerivedArgumentTag::REFERENCE; }
    [[nodiscard]] auto is_resource() const noexcept { return derived_argument_tag() == DerivedArgumentTag::RESOURCE; }
    LUISA_XIR_DEFINED_ISA_METHOD(Argument, argument)
};

template<typename Derived, DerivedArgumentTag tag>
class DerivedArgument : public Argument {
public:
    using derived_argument_type = Derived;
    using Argument::Argument;

    [[nodiscard]] static constexpr auto
    static_derived_argument_tag() noexcept { return tag; }

    [[nodiscard]] DerivedArgumentTag
    derived_argument_tag() const noexcept final { return static_derived_argument_tag(); }
};

class ValueArgument final : public DerivedArgument<ValueArgument, DerivedArgumentTag::VALUE> {
public:
    using DerivedArgument::DerivedArgument;
};

class ReferenceArgument final : public DerivedArgument<ReferenceArgument, DerivedArgumentTag::REFERENCE> {
public:
    using DerivedArgument::DerivedArgument;
};

class ResourceArgument final : public DerivedArgument<ResourceArgument, DerivedArgumentTag::RESOURCE> {
public:
    using DerivedArgument::DerivedArgument;
};

using ArgumentList = InlineIntrusiveList<Argument>;

}// namespace luisa::compute::xir
