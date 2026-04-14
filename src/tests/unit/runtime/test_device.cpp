/**
 * @file test/feat/common/test_device.cpp
 * @author sailing-innocent
 * @date 2023/07/30
 * @brief the device test suite
*/
#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>

using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

int test_wrapped_device(const char *cwd, const char *device_name) {
    return 0;
}
int test_create_device(const char *cwd, const char *device_name) {
    return 0;
}

}// namespace

static inline const auto _luisa_reg_device_create = [] {
    "device_create"_test = [] {
        auto argc = boost::ut::detail::cfg::largc;
        auto argv = boost::ut::detail::cfg::largv;
        if (argc <= 1) { return; }
        boost::ut::expect(test_create_device(argv[0], argv[1]) == 0);
    };
    return 0;
}();

static inline const auto _luisa_reg_device_wrapped = [] {
    "device_wrapped"_test = [] {
        auto argc = boost::ut::detail::cfg::largc;
        auto argv = boost::ut::detail::cfg::largv;
        if (argc <= 1) { return; }
        boost::ut::expect(test_wrapped_device(argv[0], argv[1]) == 0);
    };
    return 0;
}();

int main() {}
