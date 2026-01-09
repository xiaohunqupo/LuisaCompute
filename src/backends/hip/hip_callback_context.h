//
// Created by mike on 1/9/26.
//

#pragma once

#include <hip/hip_runtime.h>

namespace luisa::compute::hip {

struct HIPCallbackContext {
    virtual void recycle() noexcept = 0;
    virtual ~HIPCallbackContext() noexcept = default;
};

}// namespace luisa::compute::hip
