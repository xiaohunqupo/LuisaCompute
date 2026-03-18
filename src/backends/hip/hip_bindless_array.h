#pragma once

#include <hip/hip_runtime.h>

namespace luisa::compute::hip {

class HIPBindlessArray {

public:
    struct Binding {
        hipDeviceptr_t slots;
        size_t capacity;
    };

    HIPBindlessArray() noexcept = default;
    ~HIPBindlessArray() noexcept = default;
    [[nodiscard]] Binding binding() const noexcept { return {nullptr, 0}; }
};

}// namespace luisa::compute::hip
