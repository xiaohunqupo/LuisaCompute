#pragma once

#include <hip/hip_runtime.h>

namespace luisa::compute::hip {

class HIPAccel {

public:
    struct Binding {
        uint64_t handle;
        uint64_t instance_buffer;
    };

    HIPAccel() noexcept = default;
    ~HIPAccel() noexcept = default;
    [[nodiscard]] Binding binding() const noexcept { return {0, 0}; }
};

}// namespace luisa::compute::hip
