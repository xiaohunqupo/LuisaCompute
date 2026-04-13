/**
 * @file test/feat/common/test_device.cpp
 * @author sailing-innocent
 * @date 2023/07/30
 * @brief the device test suite
*/
#include "config.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>

namespace luisa::test {

class WrappedDevice {
public:
    auto device() noexcept { return m_device; }
private:
    luisa::compute::Device m_device;
};

int test_wrapped_device(luisa::string cwd, luisa::string device_name) {
    return 0;
}
int test_create_device(luisa::string cwd, luisa::string device_name) {
    return 0;
}

}// namespace luisa::test

static inline const auto _luisa_reg_device_create = [] {
    boost::ut::detail::test{"test", "device_create"} = [] {
        for (auto i = 0; i < luisa::test::backends_to_test_count(); i++) {
            luisa::string device_name = luisa::test::backends_to_test()[i];
            boost::ut::expect(static_cast<bool>(luisa::test::test_create_device(luisa::test::argv()[0], device_name) == 0));
        }
    };
    return 0;
}();

static inline const auto _luisa_reg_device_wrapped = [] {
    boost::ut::detail::test{"test", "device_wrapped"} = [] {
        for (auto i = 0; i < luisa::test::backends_to_test_count(); i++) {
            luisa::string device_name = luisa::test::backends_to_test()[i];
            boost::ut::expect(static_cast<bool>(luisa::test::test_wrapped_device(luisa::test::argv()[0], device_name) == 0));
        }
    };
    return 0;
}();
