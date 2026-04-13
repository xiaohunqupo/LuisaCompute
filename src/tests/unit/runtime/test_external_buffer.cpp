/**
 * @file test/feat/common/test_external_buffer.cpp
 * @author sailing-innocent
 * @date 2023/11/02
 * @brief test import_external_buffer
*/

#include "config.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

int test_external_buffer(Device &device) {
    constexpr uint n = 10u;
    Buffer<int> a = device.create_buffer<int>(n);
    Stream stream = device.create_stream();
    luisa::vector<int> data_init(n, 1);
    luisa::vector<int> data_result(n, 0);

    stream << a.copy_from(data_init.data());
    stream << synchronize();

    auto b = device.import_external_buffer<int>(a.native_handle(), n);

    stream << b.copy_to(data_result.data());
    stream << synchronize();

    for (uint idx = 0u; idx < n; idx++) {
        boost::ut::expect(static_cast<bool>(data_result[idx] == 1));
    }

    return 0;
}

}// namespace luisa::test

static inline const auto _luisa_reg_external_buffer = [] {
    boost::ut::detail::test{"test", "external_buffer"} = [] {
        Context context{luisa::test::argv()[0]};
        Device device = context.create_device("dx");
        boost::ut::expect(static_cast<bool>(luisa::test::test_external_buffer(device) == 0));
    };
    return 0;
}();
