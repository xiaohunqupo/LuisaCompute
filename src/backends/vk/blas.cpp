#include "blas.h"
#include "device.h"
#include "log.h"
#include "tlas.h"
#include "stream.h"
#include <luisa/runtime/rtx/aabb.h>
namespace lc::vk {
Blas::Blas(Device *device, AccelOption const &option)
    : Resource(device), option(option) {}
void Blas::_pre_build(
    CommandBuffer &cmdbuffer,
    VkAccelerationStructureGeometryKHR *acceleration_structure_geometry,
    uint32_t primitive_count,
    AccelBuildRequest request) {

    acceleration_build_geometry_info = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureBuildGeometryInfoKHR>();
    acceleration_build_geometry_info->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    acceleration_build_geometry_info->type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    acceleration_build_geometry_info->flags = option.hint == AccelOption::UsageHint::FAST_BUILD ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (option.allow_update) {
        acceleration_build_geometry_info->flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }
    acceleration_build_geometry_info->geometryCount = 1;
    acceleration_build_geometry_info->pGeometries = acceleration_structure_geometry;
    bool update = option.allow_update && request == AccelBuildRequest::PREFER_UPDATE;

    VkAccelerationStructureBuildSizesInfoKHR acceleration_structure_build_sizes_info{};
    acceleration_structure_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    acceleration_build_geometry_info->mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        device()->logic_device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        acceleration_build_geometry_info,
        &primitive_count,
        &acceleration_structure_build_sizes_info);
    uint scratch_buffer_size = update ? acceleration_structure_build_sizes_info.updateScratchSize : acceleration_structure_build_sizes_info.buildScratchSize;
    if (_accel_buffer && _accel_buffer->byte_size() != acceleration_structure_build_sizes_info.accelerationStructureSize) {
        cmdbuffer.states()->dispose_after_flush(std::move(_accel_buffer));
    }
    if (!_accel_buffer) {
        _accel_buffer = vstd::make_unique<DefaultBuffer>(
            device(),
            acceleration_structure_build_sizes_info.accelerationStructureSize,
            false, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    }
    cmdbuffer.resource_barrier->record(
        _accel_buffer.get(),
        ResourceBarrier::Usage::BuildAccel);
    VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info{};
    acceleration_structure_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    acceleration_structure_create_info.buffer = _accel_buffer->vk_buffer();
    acceleration_structure_create_info.size = acceleration_structure_build_sizes_info.accelerationStructureSize;
    acceleration_structure_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (_accel) {
        cmdbuffer.states()->_callbacks.emplace_back([a = _accel, device = device()]() {
            vkDestroyAccelerationStructureKHR(device->logic_device(), a, Device::alloc_callbacks());
        });
        sync_tlas();
    }
    VK_CHECK_RESULT(vkCreateAccelerationStructureKHR(device()->logic_device(), &acceleration_structure_create_info, Device::alloc_callbacks(), &_accel));
    scratch_buffer_size = (scratch_buffer_size + 255) & (~(255u));
    auto scratch_chunk = cmdbuffer.scratch_buffer_alloc->allocate(scratch_buffer_size);

    scratch_buffer = reinterpret_cast<Buffer const *>(scratch_chunk.handle);
    scratch_buffer_offset = scratch_chunk.offset;
    cmdbuffer.resource_barrier->record(
        scratch_buffer,
        ResourceBarrier::Usage::ComputeUAV);
}
void Blas::pre_build(
    CommandBuffer &cmdbuffer,
    ProceduralPrimitiveBuildCommand const *cmd) {
    auto acceleration_structure_geometry = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureGeometryKHR>();
    acceleration_structure_geometry->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    acceleration_structure_geometry->geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    acceleration_structure_geometry->flags = 0;
    VkDeviceOrHostAddressConstKHR aabb_data_device_address{};
    aabb_data_device_address.deviceAddress = reinterpret_cast<Buffer const *>(cmd->aabb_buffer())->get_device_address() + cmd->aabb_buffer_offset();

    auto &aabbs = acceleration_structure_geometry->geometry.aabbs;
    aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbs.data = aabb_data_device_address;
    aabbs.stride = sizeof(luisa::compute::AABB);
    _pre_build(cmdbuffer, acceleration_structure_geometry, cmd->aabb_buffer_size() / sizeof(luisa::compute::AABB), cmd->request());
    cmdbuffer.resource_barrier->record(
        reinterpret_cast<Buffer const *>(cmd->aabb_buffer()),
        ResourceBarrier::Usage::AccelInstanceBuffer);
}
void Blas::pre_build(
    CommandBuffer &cmdbuffer,
    MeshBuildCommand const *cmd) {
    VkDeviceOrHostAddressConstKHR vertex_data_device_address{};
    VkDeviceOrHostAddressConstKHR index_data_device_address{};
    vertex_data_device_address.deviceAddress = reinterpret_cast<Buffer const *>(cmd->vertex_buffer())->get_device_address() + cmd->vertex_buffer_offset();
    index_data_device_address.deviceAddress = reinterpret_cast<Buffer const *>(cmd->triangle_buffer())->get_device_address() + cmd->triangle_buffer_offset();

    auto acceleration_structure_geometry = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureGeometryKHR>();
    acceleration_structure_geometry->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    acceleration_structure_geometry->geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    acceleration_structure_geometry->flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto &triangles = acceleration_structure_geometry->geometry.triangles;
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData = vertex_data_device_address;
    triangles.maxVertex = cmd->vertex_buffer_size() / cmd->vertex_stride();
    triangles.vertexStride = cmd->vertex_stride();
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData = index_data_device_address;
    _pre_build(cmdbuffer, acceleration_structure_geometry, cmd->triangle_buffer_size() / 12, cmd->request());
    cmdbuffer.resource_barrier->record(
        reinterpret_cast<Buffer const *>(cmd->vertex_buffer()),
        ResourceBarrier::Usage::AccelInstanceBuffer);
    cmdbuffer.resource_barrier->record(
        reinterpret_cast<Buffer const *>(cmd->triangle_buffer()),
        ResourceBarrier::Usage::AccelInstanceBuffer);
}
void Blas::build(
    CommandBuffer &cmdbuffer,
    ProceduralPrimitiveBuildCommand const *cmd) {
    acceleration_build_geometry_info->dstAccelerationStructure = _accel;
    acceleration_build_geometry_info->scratchData.deviceAddress = scratch_buffer->get_device_address() + scratch_buffer_offset;
    auto acceleration_structure_build_range_info = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureBuildRangeInfoKHR>();
    acceleration_structure_build_range_info->primitiveCount = cmd->aabb_buffer_size() / sizeof(luisa::compute::AABB);
    acceleration_structure_build_range_info->primitiveOffset = 0;
    acceleration_structure_build_range_info->firstVertex = 0;
    acceleration_structure_build_range_info->transformOffset = 0;
    vkCmdBuildAccelerationStructuresKHR(
        cmdbuffer.cmdbuffer(),
        1,
        acceleration_build_geometry_info,
        &acceleration_structure_build_range_info);
}
void Blas::build(
    CommandBuffer &cmdbuffer,
    MeshBuildCommand const *cmd) {
    acceleration_build_geometry_info->dstAccelerationStructure = _accel;
    acceleration_build_geometry_info->scratchData.deviceAddress = scratch_buffer->get_device_address() + scratch_buffer_offset;
    auto acceleration_structure_build_range_info = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureBuildRangeInfoKHR>();
    acceleration_structure_build_range_info->primitiveCount = cmd->triangle_buffer_size() / 12;
    acceleration_structure_build_range_info->primitiveOffset = 0;
    acceleration_structure_build_range_info->firstVertex = 0;
    acceleration_structure_build_range_info->transformOffset = 0;
    vkCmdBuildAccelerationStructuresKHR(
        cmdbuffer.cmdbuffer(),
        1,
        acceleration_build_geometry_info,
        &acceleration_structure_build_range_info);
}
Blas::~Blas() {
    for (auto &&i : handles) {
        i->accel->allInstance[i->accelIndex].handle = nullptr;
        MeshHandle::DestroyHandle(i);
    }
    vkDestroyAccelerationStructureKHR(device()->logic_device(), _accel, Device::alloc_callbacks());
}
uint64_t Blas::get_accel_device_address() const {
    VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info{};
    acceleration_device_address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    acceleration_device_address_info.accelerationStructure = _accel;
    return vkGetAccelerationStructureDeviceAddressKHR(device()->logic_device(), &acceleration_device_address_info);
}
void Blas::remove_accel_ref(MeshHandle *handle) {
    LUISA_ASSUME(handle->mesh == this);
    {
        std::lock_guard lck(handleMtx);
        auto last = handles.back();
        handles.pop_back();
        if (last != handle) {
            last->meshIndex = handle->meshIndex;
            handles[handle->meshIndex] = last;
        }
    }
    MeshHandle::DestroyHandle(handle);
}
MeshHandle *Blas::add_accel_ref(Tlas *accel, uint index) {
    auto meshHandle = MeshHandle::AllocateHandle();
    meshHandle->mesh = this;
    meshHandle->accel = accel;
    meshHandle->accelIndex = index;
    {
        std::lock_guard lck(handleMtx);
        meshHandle->meshIndex = handles.size();
        handles.emplace_back(meshHandle);
    }
    return meshHandle;
}
void Blas::sync_tlas() {
    std::lock_guard lck(handleMtx);
    for (auto &&i : handles) {
        LUISA_ASSUME(i->mesh == this);
        i->accel->update_mesh(i);
    }
}

namespace detail {
static vstd::Pool<MeshHandle> meshHandlePool(256, false);
static vstd::spin_mutex meshHandleMtx;
}// namespace detail
MeshHandle *MeshHandle::AllocateHandle() {
    using namespace detail;
    return meshHandlePool.create_lock(meshHandleMtx);
}
void MeshHandle::DestroyHandle(MeshHandle *handle) {
    using namespace detail;
    meshHandlePool.destroy_lock(meshHandleMtx, handle);
}
}// namespace lc::vk