#include "tlas.h"
#include "stream.h"
#include "buffer.h"
#include "compute_shader.h"
#include "device.h"
#include "log.h"
#include "blas.h"
namespace lc::vk {
namespace tlas_detail {
struct TlasInputInst {
    float affine[12];
    std::array<uint, 2> mesh;
    uint index : 24;
    uint vis_mask : 8;
    uint user_id : 24;
    uint flags : 8;
};
}// namespace tlas_detail
Tlas::Tlas(Device *device, AccelOption const &option)
    : Resource(device), acceleration_build_geometry_info(nullptr) {
    if (!device->enable_raytracing()) [[unlikely]] {
        LUISA_ERROR("Raytracing not enabled, TLAS can not be loaded.");
    }
}
void Tlas::pre_build(
    CommandBuffer &cmdbuffer,
    uint instance_count,
    luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
    luisa::vector<uint4> &cache,
    luisa::span<AccelBuildCommand::Modification const> modifications,
    AccelBuildRequest request) {
    using namespace tlas_detail;
    resize_instance(instance_count);
    auto dst_inst_size = instance_count * sizeof(VkAccelerationStructureInstanceKHR);
    dst_inst_size = (dst_inst_size + 65535u) & (~65535u);
    // resize
    auto resource_barrier = cmdbuffer.resource_barrier;
    bool update = option.allow_update && request == AccelBuildRequest::PREFER_UPDATE && (!require_rebuild);
    require_rebuild = false;
    if (_last_instance_count != instance_count) {
        update = false;
        _last_instance_count = instance_count;
    }
    if (_instance_buffer && _instance_buffer->byte_size() < dst_inst_size) {
        update = false;
        auto new_inst_buffer = vstd::make_unique<DefaultBuffer>(device(), dst_inst_size, true, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
        resource_barrier->record(
            BufferView{new_inst_buffer.get()},
            ResourceBarrier::Usage::CopyDest);
        resource_barrier->record(
            BufferView{
                _instance_buffer.get()},
            ResourceBarrier::Usage::CopySource);
        resource_barrier->update_states(cmdbuffer.cmdbuffer());
        VkBufferCopy2 buffer_copy{
            VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            nullptr,
            0, 0,
            _instance_buffer->byte_size()};
        VkCopyBufferInfo2 copy_info2{
            VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
            nullptr,
            _instance_buffer->vk_buffer(),
            new_inst_buffer->vk_buffer(),
            1,
            &buffer_copy};
        vkCmdCopyBuffer2(
            cmdbuffer.cmdbuffer(),
            &copy_info2);
        cmdbuffer.states()->dispose_after_flush(std::move(_instance_buffer));
        _instance_buffer = std::move(new_inst_buffer);
    } else if (!_instance_buffer) {
        update = false;
        _instance_buffer = vstd::make_unique<DefaultBuffer>(device(), dst_inst_size, false, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    }
    if (!(modifications.empty() && set_map.empty())) {
        for (auto &&i : modifications) {
            auto ite = set_map.find(i.index);
            bool updateMesh = (i.flags & AccelBuildCommand::Modification::flag_primitive);
            if (ite != set_map.end()) {
                if (!updateMesh) {
                    const_cast<uint &>(i.flags) = i.flags | AccelBuildCommand::Modification::flag_primitive;
                    const_cast<uint64_t &>(i.primitive) = ite->second->mesh->get_accel_device_address();
                    updateMesh = true;
                }
                set_map.erase(ite);
            }
            if (updateMesh) {
                auto mesh = reinterpret_cast<Blas *>(i.primitive);
                set_mesh(mesh, i.index);
                update = false;
            }
        }
        resource_barrier->record(
            BufferView{
                _instance_buffer.get()},
            ResourceBarrier::Usage::ComputeUAV);
        auto shader = device()->set_accel_kernel.Get(device());
        resource_barrier->update_states(cmdbuffer.cmdbuffer());
        VkDescriptorSet desc_set;
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = cmdbuffer.states()->_desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = shader->desc_set_layout().data()};
        VK_CHECK_RESULT(
            vkAllocateDescriptorSets(
                device()->logic_device(),
                &alloc_info,
                &desc_set));
        const uint modification_size = modifications.size() + set_map.size();
        uint2 value = {
            modification_size,
            instance_count};
        vkCmdPushConstants(
            cmdbuffer.cmdbuffer(),
            shader->pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(value),
            &value);

        auto dsc_buffer = cmdbuffer.states()->upload_alloc.allocate((modification_size) * sizeof(TlasInputInst), 16);
        cache.clear();
        luisa::enlarge_by(cache, (dsc_buffer.size_bytes + sizeof(uint4) - 1) / sizeof(uint4));
        std::memset(cache.data(), 0, luisa::size_bytes(cache));
        auto inst_ptr = reinterpret_cast<TlasInputInst *>(cache.data());

        for (auto &&i : modifications) {
            std::memcpy(inst_ptr->affine, i.affine, sizeof(float) * 12);
            inst_ptr->index = i.index;
            inst_ptr->vis_mask = i.vis_mask;
            inst_ptr->user_id = i.user_id;
            inst_ptr->flags = i.flags;
            if (i.flags & AccelBuildCommand::Modification::flag_primitive) {
                auto mesh = reinterpret_cast<Blas const *>(i.primitive);
                auto addr = mesh->get_accel_device_address();
                inst_ptr->mesh = reinterpret_cast<std::array<uint, 2> const &>(addr);
                resource_barrier->record(BufferView{mesh->_accel_buffer.get()},
                                         ResourceBarrier::Usage::AccelInstanceBuffer);
            }
            inst_ptr++;
        }
        for (auto &i : set_map) {
            if (i.first >= allInstance.size()) continue;
            inst_ptr->index = i.first;
            inst_ptr->flags = AccelBuildCommand::Modification::flag_primitive;
            resource_barrier->record(BufferView{i.second->mesh->_accel_buffer.get()},
                                     ResourceBarrier::Usage::AccelInstanceBuffer);
            auto addr = i.second->mesh->get_accel_device_address();
            inst_ptr->mesh = reinterpret_cast<std::array<uint, 2> &>(addr);
            ++inst_ptr;
        }
        static_cast<UploadBuffer const *>(dsc_buffer.buffer)->copy_from(cache.data(), dsc_buffer.offset, dsc_buffer.size_bytes);
        set_map.clear();
        VkDescriptorBufferInfo arg_buffer_info{
            dsc_buffer.buffer->vk_buffer(),
            dsc_buffer.offset,
            dsc_buffer.size_bytes};
        VkDescriptorBufferInfo buffer_info{
            _instance_buffer->vk_buffer(),
            0,
            _instance_buffer->byte_size()};
        write_desc_sets.emplace_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            desc_set,
            0,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &arg_buffer_info,
            nullptr});
        write_desc_sets.emplace_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            desc_set,
            1,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &buffer_info,
            nullptr});
        vkUpdateDescriptorSets(
            device()->logic_device(),
            write_desc_sets.size(),
            write_desc_sets.data(),
            0,
            nullptr);
        write_desc_sets.clear();
        vkCmdBindDescriptorSets(
            cmdbuffer.cmdbuffer(),
            VK_PIPELINE_BIND_POINT_COMPUTE,
            shader->pipeline_layout(),
            0,
            1,
            &desc_set,
            0,
            nullptr);
        vkCmdBindPipeline(cmdbuffer.cmdbuffer(), VK_PIPELINE_BIND_POINT_COMPUTE, shader->pipeline());
        vkCmdDispatch(cmdbuffer.cmdbuffer(), (modification_size + 255) / 256, 1, 1);
    }
    VkDeviceOrHostAddressConstKHR instance_data_device_address{};
    instance_data_device_address.deviceAddress = _instance_buffer->get_device_address();
    auto acceleration_structure_geometry = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureGeometryKHR>();
    acceleration_structure_geometry->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    acceleration_structure_geometry->geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    acceleration_structure_geometry->flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    acceleration_structure_geometry->geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    acceleration_structure_geometry->geometry.instances.arrayOfPointers = VK_FALSE;
    acceleration_structure_geometry->geometry.instances.data = instance_data_device_address;

    acceleration_build_geometry_info = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureBuildGeometryInfoKHR>();
    acceleration_build_geometry_info->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    acceleration_build_geometry_info->type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    acceleration_build_geometry_info->flags = option.hint == AccelOption::UsageHint::FAST_BUILD ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    acceleration_build_geometry_info->geometryCount = 1;
    acceleration_build_geometry_info->pGeometries = acceleration_structure_geometry;
    if (option.allow_update) {
        acceleration_build_geometry_info->flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }
    acceleration_build_geometry_info->mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    VkAccelerationStructureBuildSizesInfoKHR acceleration_structure_build_sizes_info{};
    acceleration_structure_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        device()->logic_device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        acceleration_build_geometry_info,
        &instance_count,
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
    resource_barrier->record(
        BufferView{
            _instance_buffer.get()},
        ResourceBarrier::Usage::AccelInstanceBuffer);
    resource_barrier->record(
        _accel_buffer.get(),
        ResourceBarrier::Usage::BuildAccel);
    VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info{};
    acceleration_structure_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    acceleration_structure_create_info.buffer = _accel_buffer->vk_buffer();
    acceleration_structure_create_info.size = acceleration_structure_build_sizes_info.accelerationStructureSize;
    acceleration_structure_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    if (_accel) {
        cmdbuffer.states()->_callbacks.emplace_back([a = _accel, device = device()]() {
            vkDestroyAccelerationStructureKHR(device->logic_device(), a, Device::alloc_callbacks());
        });
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
void Tlas::update_mesh(
    MeshHandle *handle) {
    auto instIndex = handle->accelIndex;
    LUISA_ASSUME(allInstance[instIndex].handle == handle);
    set_map[instIndex] = handle;
    require_rebuild = true;
}
void Tlas::build(
    CommandBuffer &cmdbuffer,
    uint instance_count) {
    acceleration_build_geometry_info->dstAccelerationStructure = _accel;
    acceleration_build_geometry_info->scratchData.deviceAddress = scratch_buffer->get_device_address() + scratch_buffer_offset;
    auto acceleration_structure_build_range_info = cmdbuffer.temp_desc->allocate_memory<VkAccelerationStructureBuildRangeInfoKHR>();
    acceleration_structure_build_range_info->primitiveCount = instance_count;
    acceleration_structure_build_range_info->primitiveOffset = 0;
    acceleration_structure_build_range_info->firstVertex = 0;
    acceleration_structure_build_range_info->transformOffset = 0;
    vkCmdBuildAccelerationStructuresKHR(
        cmdbuffer.cmdbuffer(),
        1,
        acceleration_build_geometry_info,
        &acceleration_structure_build_range_info);
}
void Tlas::resize_instance(size_t size) {
    if (size < allInstance.size()) {
        for (auto &i : vstd::ptr_range(allInstance.data() + size, allInstance.data() + allInstance.size())) {
            if (!i.handle) continue;
            i.handle->mesh->remove_accel_ref(i.handle);
        }
    }
    allInstance.resize(size);
}

Tlas::~Tlas() {
    for (auto &&i : allInstance) {
        auto mesh = i.handle;
        if (mesh)
            mesh->mesh->remove_accel_ref(mesh);
    }
    vkDestroyAccelerationStructureKHR(device()->logic_device(), _accel, Device::alloc_callbacks());
}
void Tlas::set_mesh(Blas *mesh, uint64 index) {
    auto &&inst = allInstance[index].handle;
    if (inst != nullptr) {
        if (inst->mesh == mesh) return;
        inst->mesh->remove_accel_ref(inst);
    }
    inst = mesh->add_accel_ref(this, index);
    inst->accelIndex = index;
}
}// namespace lc::vk