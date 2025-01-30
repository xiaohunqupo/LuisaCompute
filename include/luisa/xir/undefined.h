#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class LC_XIR_API Undefined : public DerivedValue<DerivedValueTag::UNDEFINED> {
public:
    using DerivedValue::DerivedValue;
    [[nodiscard]] static Undefined *create(const Type *type) noexcept;
};

}// namespace luisa::compute::xir
