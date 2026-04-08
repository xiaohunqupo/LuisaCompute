//
// Created by mike on 4/8/26.
//

#include "hip_check.h"
#include "hip_buffer.h"
#include "hip_command_encoder.h"
#include "hip_stream.h"
#include "hip_device.h"
#include "hip_procedural_primitive.h"

namespace luisa::compute::hip {

HIPProceduralPrimitive::HIPProceduralPrimitive(hiprtContext ctx, const AccelOption &option) noexcept
    : _option{option}, _hiprt_ctx{ctx} {}

HIPProceduralPrimitive::~HIPProceduralPrimitive() noexcept {
    if (_geometry) {
        hiprtDestroyGeometry(_hiprt_ctx, _geometry);
    }
}

void HIPProceduralPrimitive::build(HIPCommandEncoder &encoder, ProceduralPrimitiveBuildCommand *command) noexcept {

    auto aabb_buffer = reinterpret_cast<const HIPBuffer *>(command->aabb_buffer());
    LUISA_ASSERT(command->aabb_buffer_offset() + command->aabb_buffer_size() <= aabb_buffer->size_bytes(),
                 "AABB buffer offset + size exceeds buffer size {}.", aabb_buffer->size_bytes());

    std::scoped_lock lock{_mutex};

    auto new_aabb_buffer = static_cast<std::byte *>(aabb_buffer->handle()) + command->aabb_buffer_offset();
    auto new_aabb_buffer_size = command->aabb_buffer_size();

    auto requires_build =
        _geometry == nullptr ||
        !_option.allow_update ||
        command->request() == AccelBuildRequest::FORCE_BUILD ||
        reinterpret_cast<hipDeviceptr_t>(new_aabb_buffer) != _aabb_buffer ||
        new_aabb_buffer_size != _aabb_buffer_size;

    _aabb_buffer = reinterpret_cast<hipDeviceptr_t>(new_aabb_buffer);
    _aabb_buffer_size = new_aabb_buffer_size;

    auto aabb_count = static_cast<uint32_t>(_aabb_buffer_size / sizeof(AABB));

    hiprtAABBListPrimitive aabb_prim{};
    aabb_prim.aabbs = reinterpret_cast<hiprtDevicePtr>(_aabb_buffer);
    aabb_prim.aabbCount = aabb_count;
    aabb_prim.aabbStride = sizeof(AABB);

    hiprtGeometryBuildInput build_input{};
    build_input.type = hiprtPrimitiveTypeAABBList;
    build_input.primitive.aabbList = aabb_prim;

    hiprtBuildOptions build_options{};
    build_options.buildFlags = hiprtBuildFlagBitPreferHighQualityBuild;

    auto hip_stream = encoder.stream()->handle();

    if (requires_build) {
        if (_geometry) {
            hiprtDestroyGeometry(_hiprt_ctx, _geometry);
            _geometry = nullptr;
        }
        LUISA_CHECK_HIPRT(hiprtCreateGeometry(_hiprt_ctx, build_input, build_options, _geometry));

        size_t temp_size = 0;
        LUISA_CHECK_HIPRT(hiprtGetGeometryBuildTemporaryBufferSize(_hiprt_ctx, build_input, build_options, temp_size));

        hipDeviceptr_t temp_buffer{};
        if (temp_size > 0) {
            LUISA_CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&temp_buffer), temp_size, hip_stream));
        }

        LUISA_CHECK_HIPRT(hiprtBuildGeometry(_hiprt_ctx, hiprtBuildOperationBuild,
                                             build_input, build_options,
                                             temp_buffer, hip_stream, _geometry));

        if (temp_buffer) {
            LUISA_CHECK_HIP(hipFreeAsync(reinterpret_cast<void *>(temp_buffer), hip_stream));
        }
    } else {
        LUISA_CHECK_HIPRT(hiprtBuildGeometry(_hiprt_ctx, hiprtBuildOperationUpdate,
                                             build_input, build_options,
                                             nullptr, hip_stream, _geometry));
    }
}

}// namespace luisa::compute::hip
