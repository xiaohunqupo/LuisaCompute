/**
 * @file test_gemm.cpp
 * @brief The GEMM TestSuite
 * @author sailing-innocent
 * @date 2025-04-23
 */

#include "common/config.h"
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

bool test_tensor_gemm(Device &device) {
    return true;
}

}// namespace luisa::test

TEST_SUITE("tensor") {
    LUISA_TEST_CASE_WITH_DEVICE("tensor_gemm", luisa::test::test_tensor_gemm(device));
}
