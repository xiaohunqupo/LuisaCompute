#pragma once
#include "resource.h"
namespace lc::vk {
// Base class for all primitive types that can be placed in a TLAS.
// This allows the TLAS to distinguish between Blas (mesh/procedural) and MotionInstance.
class PrimitiveBase : public Resource {
public:
    enum class PrimTag {
        BLAS,
        MOTION_INSTANCE
    };

private:
    PrimTag _prim_tag;

public:
    PrimitiveBase(Device *device, PrimTag prim_tag) : Resource(device), _prim_tag(prim_tag) {}
    [[nodiscard]] auto prim_tag() const noexcept { return _prim_tag; }
    [[nodiscard]] bool is_motion_instance() const noexcept { return _prim_tag == PrimTag::MOTION_INSTANCE; }
};
}// namespace lc::vk
