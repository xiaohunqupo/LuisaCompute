#pragma once

#include "ut/ut.hpp"

namespace luisa::test {

[[nodiscard]] int argc() noexcept;
[[nodiscard]] const char *const *argv() noexcept;
[[nodiscard]] int backends_to_test_count() noexcept;
[[nodiscard]] const char *const *backends_to_test() noexcept;

}// namespace luisa::test

#include <luisa/core/stl/string.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>

// `name` must be a string literal. `condition` may reference `device` (luisa::compute::Device&).
#define LUISA_TEST_IMPL_CONCAT2_(a, b) a##b
#define LUISA_TEST_IMPL_CONCAT_(a, b) LUISA_TEST_IMPL_CONCAT2_(a, b)
#define LUISA_TEST_CASE_WITH_DEVICE(name, condition)                                        \
    static inline const auto LUISA_TEST_IMPL_CONCAT_(_luisa_reg_, __COUNTER__) = [] {       \
        using namespace boost::ut;                                                          \
        using namespace boost::ut::literals;                                                \
        boost::ut::detail::test{"test", name} = [&] {                                       \
            for (auto i = 0; i < luisa::test::backends_to_test_count(); i++) {              \
                luisa::string device_name = luisa::test::backends_to_test()[i];             \
                luisa::compute::Context context{luisa::test::argv()[0]};                    \
                luisa::compute::Device device = context.create_device(device_name.c_str()); \
                boost::ut::expect(static_cast<bool>(condition))                             \
                    << "failed on backend:" << device_name.c_str();                         \
            }                                                                               \
        };                                                                                  \
        return 0;                                                                           \
    }()
