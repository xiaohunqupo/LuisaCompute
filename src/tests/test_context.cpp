// Test for compute context and backend enumeration.
// This test verifies that the compute context can properly discover and report
// available backends and their associated devices.

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include "common/config.h"

TEST_CASE("context") {
    // Initialize context with the test executable path
    luisa::compute::Context context{luisa::test::argv()[0]};
    
    // Iterate over all installed backends
    for (auto &&backend : context.installed_backends()) {
        // Get list of device names for this backend
        auto device_names = context.backend_device_names(backend);
        
        // Verify that at least one device is available
        REQUIRE(!device_names.empty());
        
        // Log each discovered device
        for (auto &device_name : device_names) {
            LUISA_INFO("Found device '{}' for backend '{}'.",
                       device_name, backend);
        }
    }
}
