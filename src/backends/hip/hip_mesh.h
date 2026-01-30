//
// Created by mike on 1/30/26.
//

#pragma once

#include <luisa/runtime/rtx/mesh.h>
#include <hiprt/hiprt.h>

namespace luisa::compute::hip {

class HIPMesh {

private:
    hiprtGeometry _geometry{nullptr};

public:
};

}// namespace luisa::compute::hip
