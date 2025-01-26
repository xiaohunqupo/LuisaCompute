//
// Created by Mike Smith on 2022/2/11.
//

#include <luisa/core/stl.h>
#include <luisa/core/logging.h>

#include "fallback_mesh.h"
#include "fallback_accel.h"
#include "fallback_command_queue.h"

namespace luisa::compute::fallback {

FallbackAccel::FallbackAccel(RTCDevice device, const AccelOption &option) noexcept
    : _handle{rtcNewScene(device)} { luisa_fallback_accel_set_flags(_handle, option); }

FallbackAccel::~FallbackAccel() noexcept { rtcReleaseScene(_handle); }

void FallbackAccel::build(luisa::unique_ptr<AccelBuildCommand> cmd) noexcept {
    LUISA_ASSERT(!cmd->update_instance_buffer_only(),
                 "FallbackAccel does not support update_instance_buffer_only.");
    if (auto n = cmd->instance_count(); n < _instances.size()) {
        // remove redundant geometries
        for (auto i = n; i < _instances.size(); i++) { rtcDetachGeometry(_handle, i); }
        _instances.resize(n);
    } else {
        // create new geometries
        auto device = rtcGetSceneDevice(_handle);
        _instances.reserve(next_pow2(n));
        for (auto i = _instances.size(); i < n; i++) {
            auto geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
            rtcSetGeometryBuildQuality(geometry, RTC_BUILD_QUALITY_HIGH);
            rtcAttachGeometry(_handle, geometry);
            rtcReleaseGeometry(geometry);// already moved into the scene
            auto &instance = _instances.emplace_back();
            instance.geometry = geometry;
        }
    }
    for (auto m : cmd->modifications()) {
        using Mod = AccelBuildCommand::Modification;
        if (m.flags & Mod::flag_primitive) {
            auto geometry = _instances[m.index].geometry;
            auto mesh = reinterpret_cast<const FallbackMesh *>(m.primitive);
            rtcSetGeometryInstancedScene(geometry, mesh->handle());
        }
        if (m.flags & Mod::flag_transform) {
            std::memcpy(_instances[m.index].affine, m.affine, sizeof(m.affine));
        }
        if (m.flags & Mod::flag_visibility) {
            _instances[m.index].mask = m.vis_mask;
        }
        if (m.flags & Mod::flag_user_id) {
            _instances[m.index].user_id = m.user_id;
        }
        if (m.flags & Mod::flag_opaque) {
            _instances[m.index].opaque = m.flags & Mod::flag_opaque_on;
        }
        _instances[m.index].dirty = true;
    }
    for (auto &&instance : _instances) {
        if (instance.dirty) {
            auto geometry = instance.geometry;
            rtcSetGeometryTransform(geometry, 0u, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, instance.affine);
            rtcSetGeometryMask(geometry, instance.mask);
            rtcSetGeometryUserData(geometry, instance.opaque ? reinterpret_cast<void *>(1ull) : nullptr);
            rtcCommitGeometry(geometry);
            instance.dirty = false;
        }
    }
    // TODO: support update?
    rtcCommitScene(_handle);
}

}// namespace luisa::compute::fallback
