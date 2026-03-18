#pragma once

#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute::hip {

class HIPCommandEncoder;

class HIPShaderPrinter {

public:
    struct Binding {
        size_t capacity;
        void *content;
    };

    explicit HIPShaderPrinter() noexcept = default;
    ~HIPShaderPrinter() noexcept = default;

    [[nodiscard]] static luisa::unique_ptr<HIPShaderPrinter> create() noexcept {
        return luisa::make_unique<HIPShaderPrinter>();
    }

    [[nodiscard]] Binding encode(HIPCommandEncoder &encoder) const noexcept {
        return {0, nullptr};
    }
};

}// namespace luisa::compute::hip
