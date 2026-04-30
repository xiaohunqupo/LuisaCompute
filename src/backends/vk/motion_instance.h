#pragma once
#include <volk.h>
#include "primitive_base.h"
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/motion_transform.h>
namespace lc::vk {
class Blas;
using namespace luisa;
using namespace luisa::compute;

// MotionInstance wraps a child BLAS with per-keyframe transforms for motion blur.
// Uses VK_NV_ray_tracing_motion_blur extension for hardware-accelerated motion blur.
class MotionInstance : public PrimitiveBase {
    AccelMotionOption _option;
    Blas *_child{nullptr};
    luisa::vector<MotionInstanceTransform> _keyframes;

public:
    MotionInstance(Device *device, const AccelMotionOption &option);
    ~MotionInstance() override = default;

    void set_child(Blas *child) noexcept { _child = child; }
    void set_keyframes(luisa::vector<MotionInstanceTransform> keyframes) noexcept {
        _keyframes = std::move(keyframes);
    }

    [[nodiscard]] auto child() const noexcept { return _child; }
    [[nodiscard]] auto &option() const noexcept { return _option; }
    [[nodiscard]] auto &keyframes() const noexcept { return _keyframes; }
    [[nodiscard]] auto keyframe_count() const noexcept { return _option.keyframe_count; }
    [[nodiscard]] auto mode() const noexcept { return _option.mode; }
};
}// namespace lc::vk
