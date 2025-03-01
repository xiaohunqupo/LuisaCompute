//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <luisa/core/stl.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/dsl/struct.h>
#include <luisa/runtime/rtx/ray.h>
#include <luisa/runtime/rtx/hit.h>

#include "fallback_embree.h"

namespace luisa {
class ThreadPool;
}// namespace luisa

namespace luisa::compute::fallback {

class FallbackCommandQueue;
class FallbackMesh;

class alignas(16) FallbackAccel {

public:
    struct alignas(16) Instance {
        float affine[12];
        uint8_t mask;
        bool opaque;
        bool dirty;
        bool is_curve;
        uint user_id;
        RTCGeometry geometry;
    };
    static_assert(sizeof(Instance) == 64u);

    struct alignas(16) View {
        RTCScene scene;
        Instance *instances;
    };

private:
    RTCScene _handle;
    luisa::vector<Instance> _instances;

public:
    [[nodiscard]] RTCScene handle() const noexcept { return _handle; }
    FallbackAccel(RTCDevice device, const AccelOption &option) noexcept;
    ~FallbackAccel() noexcept;
    void build(luisa::unique_ptr<AccelBuildCommand> cmd) noexcept;
    [[nodiscard]] auto view() noexcept { return View{_handle, _instances.data()}; }
};

using FallbackAccelView = FallbackAccel::View;

}// namespace luisa::compute::fallback
