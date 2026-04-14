/**
 * @file test_gemm.cpp
 * @brief The GEMM TestSuite
 * @author sailing-innocent
 * @date 2025-04-23
 */

#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

bool test_tensor_gemm(Device &device) {
    return true;
}

}// namespace

static inline const auto _luisa_reg_tensor_gemm = [] {
    "tensor_gemm"_test = [] {
        auto dc = luisa::test::create_device_from_ut();
        if (!dc) { return; }
        auto &device = dc->device;
        boost::ut::expect(test_tensor_gemm(device));
    };
    return 0;
}();

int main() {}
