// Test for compute context and backend enumeration.
// This test verifies that the compute context can properly discover and report
// available backends and their associated devices.

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include "config.h"

namespace luisa::test {

[[nodiscard]] int argc() noexcept { return boost::ut::detail::cfg::largc; }
[[nodiscard]] const char *const *argv() noexcept { return boost::ut::detail::cfg::largv; }
[[nodiscard]] int backends_to_test_count() noexcept { return 0; }
[[nodiscard]] const char *const *backends_to_test() noexcept { return nullptr; }

}// namespace luisa::test

static inline const auto _luisa_reg_context = [] {
    boost::ut::detail::test{"test", "context"} = [] {
        // Initialize context with the test executable path
        luisa::compute::Context context{luisa::test::argv()[0]};

        // Iterate over all installed backends
        for (auto &&backend : context.installed_backends()) {
            // Get list of device names for this backend
            auto device_names = context.backend_device_names(backend);

            // Verify that at least one device is available
            boost::ut::expect(static_cast<bool>(!device_names.empty()));

            // Log each discovered device
            for (auto &device_name : device_names) {
                LUISA_INFO("Found device '{}' for backend '{}'.",
                           device_name, backend);
            }
        }
    };
    return 0;
}();

int main() {}
