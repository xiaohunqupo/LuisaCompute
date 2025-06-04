#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class Undefined : public DerivedGlobalValue<Undefined, DerivedValueTag::UNDEFINED> {
public:
    Undefined(Module *module, const Type *type) noexcept
        : Super{module, type} {}
};

class SentinelUndefined final : public Undefined {
public:
    explicit SentinelUndefined(Module *module) noexcept
        : Undefined{module, nullptr} {}
};

using UndefinedList = ManagedIntrusiveList<Undefined, SentinelUndefined>;

}// namespace luisa::compute::xir
