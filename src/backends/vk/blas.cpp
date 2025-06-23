#include "blas.h"
#include "device.h"
#include "log.h"
#include "stream.h"
namespace lc::vk {
Blas::Blas(Device *device, AccelOption const &option)
    : Resource(device), option(option) {}
void Blas::pre_build(
    CommandBuffer &cmdbuffer,
    MeshBuildCommand const *cmd) {
    VkDeviceOrHostAddressConstKHR vertex_data_device_address{};
    VkDeviceOrHostAddressConstKHR index_data_device_address{};
    vertex_data_device_address.deviceAddress = reinterpret_cast<DefaultBuffer const *>(cmd->vertex_buffer())->get_device_address() + cmd->vertex_buffer_offset();
    index_data_device_address.deviceAddress = reinterpret_cast<DefaultBuffer const *>(cmd->triangle_buffer())->get_device_address() + cmd->triangle_buffer_offset();

    VkTransformMatrixKHR transform_matrix = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f};
    auto acceleration_structure_geometry = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureGeometryKHR>();
    acceleration_structure_geometry->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    acceleration_structure_geometry->geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    acceleration_structure_geometry->flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    acceleration_structure_geometry->geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    acceleration_structure_geometry->geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    acceleration_structure_geometry->geometry.triangles.vertexData = vertex_data_device_address;
    acceleration_structure_geometry->geometry.triangles.maxVertex = cmd->vertex_buffer_size() / cmd->vertex_stride();
    acceleration_structure_geometry->geometry.triangles.vertexStride = cmd->vertex_stride();
    acceleration_structure_geometry->geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    acceleration_structure_geometry->geometry.triangles.indexData = index_data_device_address;

    acceleration_build_geometry_info = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureBuildGeometryInfoKHR>();
    acceleration_build_geometry_info->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    acceleration_build_geometry_info->type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    acceleration_build_geometry_info->flags = option.hint == AccelOption::UsageHint::FAST_BUILD ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (option.allow_update) {
        acceleration_build_geometry_info->flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }
    acceleration_build_geometry_info->geometryCount = 1;
    acceleration_build_geometry_info->pGeometries = acceleration_structure_geometry;
    bool update = option.allow_update && cmd->request() == AccelBuildRequest::PREFER_UPDATE;

    VkAccelerationStructureBuildSizesInfoKHR acceleration_structure_build_sizes_info{};
    acceleration_structure_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    acceleration_build_geometry_info->mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    const uint32_t primitive_count = cmd->triangle_buffer_size() / 12;
    device()->func_table.vkGetAccelerationStructureBuildSizesKHR(
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
            true);
    }
    cmdbuffer.resource_barrier->record(
        _accel_buffer.get(),
        ResourceBarrier::Usage::BuildAccel);
    VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info{};
    acceleration_structure_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    acceleration_structure_create_info.buffer = _accel_buffer->vk_buffer();
    acceleration_structure_create_info.size = acceleration_structure_build_sizes_info.accelerationStructureSize;
    acceleration_structure_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VK_CHECK_RESULT(device()->func_table.vkCreateAccelerationStructureKHR(device()->logic_device(), &acceleration_structure_create_info, Device::alloc_callbacks(), &_accel));
    scratch_buffer_size = (scratch_buffer_size + 255) & (~(255u));
    auto scratch_chunk = cmdbuffer.scratch_buffer_alloc->allocate(scratch_buffer_size);
    
    scratch_buffer = reinterpret_cast<DefaultBuffer const *>(scratch_chunk.handle);
    scratch_buffer_offset = scratch_chunk.offset;
    cmdbuffer.resource_barrier->record(
        scratch_buffer,
        ResourceBarrier::Usage::ComputeUAV);
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
    device()->func_table.vkCmdBuildAccelerationStructuresKHR(
        cmdbuffer.cmdbuffer(),
        1,
        acceleration_build_geometry_info,
        &acceleration_structure_build_range_info);
}
Blas::~Blas() {
    device()->func_table.vkDestroyAccelerationStructureKHR(device()->logic_device(), _accel, Device::alloc_callbacks());
}
}// namespace lc::vk