//
// Created by mike on 3/19/26.
//

#include "hip_check.h"
#include "hip_mesh.h"
#include "hip_command_encoder.h"
#include "hip_stream.h"
#include "hip_device.h"
#include "hip_accel.h"

namespace luisa::compute::hip {

HIPAccel::HIPAccel(hiprtContext ctx, const AccelOption &option) noexcept
    : _option{option}, _hiprt_ctx{ctx} {}

HIPAccel::~HIPAccel() noexcept {
    if (_scene) {
        hiprtDestroyScene(_hiprt_ctx, _scene);
    }
    if (_instance_buffer) {
        LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_instance_buffer)));
    }
}

void HIPAccel::_build(HIPCommandEncoder &encoder) noexcept {
    auto hip_stream = encoder.stream()->handle();
    auto n = static_cast<uint32_t>(_instance_count);

    if (_scene) {
        hiprtDestroyScene(_hiprt_ctx, _scene);
        _scene = nullptr;
    }

    hiprtSceneBuildInput build_input{};
    build_input.instanceCount = n;
    build_input.frameCount = n;
    build_input.frameType = hiprtFrameTypeMatrix;

    hipDeviceptr_t d_instances{};
    hipDeviceptr_t d_frames{};

    LUISA_CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&d_instances),
                                   n * sizeof(hiprtInstance), hip_stream));
    LUISA_CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&d_frames),
                                   n * sizeof(hiprtFrameMatrix), hip_stream));

    LUISA_CHECK_HIP(hipMemcpyHtoDAsync(d_instances, _hiprt_instances.data(),
                                       n * sizeof(hiprtInstance), hip_stream));
    LUISA_CHECK_HIP(hipMemcpyHtoDAsync(d_frames, _hiprt_frames.data(),
                                       n * sizeof(hiprtFrameMatrix), hip_stream));

    build_input.instances = reinterpret_cast<hiprtDevicePtr>(d_instances);
    build_input.instanceFrames = reinterpret_cast<hiprtDevicePtr>(d_frames);
    build_input.instanceTransformHeaders = nullptr;
    build_input.instanceMasks = nullptr;

    hiprtBuildOptions build_options{};
    build_options.buildFlags = hiprtBuildFlagBitPreferFastBuild;

    LUISA_CHECK_HIPRT(hiprtCreateScene(_hiprt_ctx, build_input, build_options, _scene));

    size_t temp_size = 0;
    LUISA_CHECK_HIPRT(hiprtGetSceneBuildTemporaryBufferSize(_hiprt_ctx, build_input, build_options, temp_size));

    hipDeviceptr_t temp_buffer{};
    if (temp_size > 0) {
        LUISA_CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&temp_buffer), temp_size, hip_stream));
    }

    LUISA_CHECK_HIPRT(hiprtBuildScene(_hiprt_ctx, hiprtBuildOperationBuild,
                                      build_input, build_options,
                                      temp_buffer, 0, _scene));

    if (temp_buffer) {
        LUISA_CHECK_HIP(hipFreeAsync(reinterpret_cast<void *>(temp_buffer), hip_stream));
    }

    LUISA_CHECK_HIP(hipFreeAsync(reinterpret_cast<void *>(d_instances), hip_stream));
    LUISA_CHECK_HIP(hipFreeAsync(reinterpret_cast<void *>(d_frames), hip_stream));
}

void HIPAccel::_update(HIPCommandEncoder &encoder) noexcept {
    auto hip_stream = encoder.stream()->handle();
    auto n = static_cast<uint32_t>(_instance_count);

    hiprtSceneBuildInput build_input{};
    build_input.instanceCount = n;
    build_input.frameCount = n;
    build_input.frameType = hiprtFrameTypeMatrix;

    hipDeviceptr_t d_instances{};
    hipDeviceptr_t d_frames{};

    LUISA_CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&d_instances),
                                   n * sizeof(hiprtInstance), hip_stream));
    LUISA_CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&d_frames),
                                   n * sizeof(hiprtFrameMatrix), hip_stream));

    LUISA_CHECK_HIP(hipMemcpyHtoDAsync(d_instances, _hiprt_instances.data(),
                                       n * sizeof(hiprtInstance), hip_stream));
    LUISA_CHECK_HIP(hipMemcpyHtoDAsync(d_frames, _hiprt_frames.data(),
                                       n * sizeof(hiprtFrameMatrix), hip_stream));

    build_input.instances = reinterpret_cast<hiprtDevicePtr>(d_instances);
    build_input.instanceFrames = reinterpret_cast<hiprtDevicePtr>(d_frames);
    build_input.instanceTransformHeaders = nullptr;
    build_input.instanceMasks = nullptr;

    hiprtBuildOptions build_options{};
    build_options.buildFlags = hiprtBuildFlagBitPreferFastBuild;

    LUISA_CHECK_HIPRT(hiprtBuildScene(_hiprt_ctx, hiprtBuildOperationUpdate,
                                      build_input, build_options,
                                      nullptr, 0, _scene));

    LUISA_CHECK_HIP(hipFreeAsync(reinterpret_cast<void *>(d_instances), hip_stream));
    LUISA_CHECK_HIP(hipFreeAsync(reinterpret_cast<void *>(d_frames), hip_stream));
}

void HIPAccel::build(HIPCommandEncoder &encoder, AccelBuildCommand *command) noexcept {

    std::scoped_lock lock{_mutex};

    auto hip_stream = encoder.stream()->handle();
    auto instance_count = command->instance_count();
    LUISA_ASSERT(instance_count > 0u, "Instance count must be greater than 0.");

    auto instance_count_changed = _instance_count != instance_count;
    if (instance_count_changed) {
        _instance_count = instance_count;
        _host_instances.resize(instance_count);
        _hiprt_instances.resize(instance_count);
        _hiprt_frames.resize(instance_count);
        _requires_rebuild = true;
    }

    auto required_size = instance_count * sizeof(CodegenInstance);
    if (_instance_buffer_size < required_size) {
        if (_instance_buffer) {
            LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_instance_buffer)));
        }
        auto alloc_size = required_size;
        LUISA_CHECK_HIP(hipMalloc(reinterpret_cast<void **>(&_instance_buffer), alloc_size));
        _instance_buffer_size = alloc_size;
    }

    auto mods = command->modifications();
    for (auto &m : mods) {
        auto idx = m.index;
        LUISA_ASSERT(idx < instance_count, "Modification index out of range.");
        auto &inst = _host_instances[idx];
        auto &hiprt_inst = _hiprt_instances[idx];
        auto &hiprt_frame = _hiprt_frames[idx];

        if (m.flags & AccelBuildCommand::Modification::flag_primitive) {
            _requires_rebuild = true;
            auto mesh = reinterpret_cast<const HIPMesh *>(m.primitive);
            auto geom_handle = mesh->handle();
            hiprt_inst.type = hiprtInstanceTypeGeometry;
            hiprt_inst.geometry = geom_handle;
            inst.mesh_handle = reinterpret_cast<uint64_t>(geom_handle);
        }

        if (m.flags & AccelBuildCommand::Modification::flag_transform) {
            // Modification::affine[12] is row-major 3x4, same layout as hiprtFrameMatrix::matrix[3][4]
            std::memcpy(hiprt_frame.matrix, m.affine, 12 * sizeof(float));
            hiprt_frame.time = 0.0f;
            std::memcpy(inst.affine, m.affine, 12 * sizeof(float));
        }

        if (m.flags & AccelBuildCommand::Modification::flag_visibility) {
            inst.visibility_mask = m.vis_mask;
        }

        if (m.flags & AccelBuildCommand::Modification::flag_user_id) {
            inst.user_id = m.user_id;
        }

        if (m.flags & AccelBuildCommand::Modification::flag_opaque_on) {
            inst.flags |= 1u;
        } else if (m.flags & AccelBuildCommand::Modification::flag_opaque_off) {
            inst.flags &= ~1u;
        }
    }

    LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
        _instance_buffer, _host_instances.data(),
        instance_count * sizeof(CodegenInstance), hip_stream));

    _requires_rebuild = _requires_rebuild ||
                        command->request() == AccelBuildRequest::FORCE_BUILD ||
                        !_option.allow_update ||
                        _scene == nullptr;

    if (!command->update_instance_buffer_only()) {
        if (_requires_rebuild) {
            _build(encoder);
        } else {
            _update(encoder);
        }
        _requires_rebuild = false;
    }
}

HIPAccel::Binding HIPAccel::binding() const noexcept {
    std::scoped_lock lock{_mutex};
    return Binding{
        reinterpret_cast<uint64_t>(_scene),
        reinterpret_cast<uint64_t>(_instance_buffer)};
}

}// namespace luisa::compute::hip
