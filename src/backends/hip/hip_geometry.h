//
// Created by mike on 4/8/26.
//

#pragma once

#include <hiprt/hiprt.h>

namespace luisa::compute::hip {

/// Common base for HIPMesh and HIPProceduralPrimitive so that
/// hip_accel.cpp can obtain the hiprtGeometry handle without
/// knowing which concrete type was emplaced into the scene.
class HIPGeometry {
public:
    virtual ~HIPGeometry() noexcept = default;
    [[nodiscard]] virtual hiprtGeometry handle() const noexcept = 0;
};

}// namespace luisa::compute::hip
