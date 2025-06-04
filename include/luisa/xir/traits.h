#pragma once

#include <luisa/core/dll_export.h>

#define LUISA_XIR_DEFINED_ISA_METHOD(BaseType, base_name)                                             \
    template<typename Derived>                                                                        \
        requires std::derived_from<Derived, BaseType>                                                 \
    [[nodiscard]] bool isa() const noexcept {                                                         \
        if constexpr (std::is_same_v<BaseType, Derived>) {                                            \
            return true;                                                                              \
        } else {                                                                                      \
            using ImmediateDerived = typename Derived::derived_##base_name##_type;                    \
            return derived_##base_name##_tag() == Derived::static_derived_##base_name##_tag() &&      \
                   (std::is_final_v<ImmediateDerived> || std::is_same_v<ImmediateDerived, Derived> || \
                    static_cast<const ImmediateDerived *>(this)->template isa<Derived>());            \
        }                                                                                             \
    }
