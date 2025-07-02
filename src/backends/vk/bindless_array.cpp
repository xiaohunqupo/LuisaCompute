#include "bindless_array.h"
#include "compute_shader.h"
#include "upload_buffer.h"
#include "device.h"
#include "resource_barrier.h"
#include "stream.h"
#include <vulkan/vulkan_core.h>
#include "log.h"
#include "device.h"
namespace lc::vk {
namespace bdls_detail {
Device::HeapAlloc &get_alloc(Device &device, BindlessType type) {
    switch (type) {
        case BindlessType::Buffer:
            return device.buffer_heap_pool;
        case BindlessType::Texture2D:
            return device.tex2d_heap_pool;
        case BindlessType::Texture3D:
            return device.tex3d_heap_pool;
        default:
            LUISA_ERROR("Bad bindless type.");
    }
}
}// namespace bdls_detail
BindlessArray::BindlessArray(Device *device, BindlessType type, size_t size)
    : Resource(device),
      _indices_buffer(device, type == BindlessType::None ? sizeof(BindlessStruct) * size : sizeof(uint)),
      _type(type) {
    switch (type) {
        case BindlessType::None:
            typed_binded.reset_as(0, size);
            break;
        default: {
            auto &alloc = bdls_detail::get_alloc(*device, type);
            _buffer_node = alloc.sub_alloc(size);
            typed_binded.reset_as(1);
            auto &v = typed_binded.get<1>();
            v.resize(size);
        } break;
    }
}
BindlessArray::~BindlessArray() {
    if (_buffer_node) {
        auto &alloc = bdls_detail::get_alloc(*device(), _type);
        alloc.free(_buffer_node);
    }
    if (typed_binded.index() == 0) {
        auto &binded = typed_binded.get<0>();
        for (auto &idx : binded) {
            auto &i = idx.first;
            if (i.buffer != BindlessStruct::n_pos) {
                device()->buffer_heap_pool.dealloc(i.buffer);
            }
            if (i.tex2D != BindlessStruct::n_pos) {
                device()->tex2d_heap_pool.dealloc(i.tex2D);
            }
            if (i.tex3D != BindlessStruct::n_pos) {
                device()->tex3d_heap_pool.dealloc(i.tex3D);
            }
        }
    }
    for (auto &i : freeQueue) {
        switch (i.type) {
            case 0:
                device()->buffer_heap_pool.dealloc(i.index);
                break;
            case 1:
                device()->tex2d_heap_pool.dealloc(i.index);
                break;
            case 2:
                device()->tex3d_heap_pool.dealloc(i.index);
                break;
        }
    }
}
void BindlessArray::pre_update(ResourceBarrier *barrier) {
    if (offset_setted && _buffer_node) return;
    barrier->record(
        BufferView{&_indices_buffer},
        _buffer_node ? ResourceBarrier::Usage::CopyDest : ResourceBarrier::Usage::ComputeUAV);
}
void BindlessArray::return_value(MapIndex &index, uint type, uint &originValue) {
    if (originValue != BindlessStruct::n_pos) {
        freeQueue.push_back(FreeValue{
            .type = type,
            .index = originValue});
        originValue = BindlessStruct::n_pos;
        auto &&v = index.value();
        v--;
        if (v == 0) {
            ptrMap.remove(index);
        }
    }
    index = {};
}
void BindlessArray::deref(Map::Index &index) {
    if (!index) return;
    auto &&v = index.value();
    v--;
    if (v == 0) {
        ptrMap.remove(index);
    }
    index = {};
}
void BindlessArray::bind(vstd::span<const BindlessArrayUpdateCommand::Texture2DModification> mods) {
    LUISA_DEBUG_ASSERT(typed_binded.index() == 1);
    auto &binded = typed_binded.get<1>();
    std::lock_guard lck{mtx};
    if (mods.empty()) return;
    for (auto &&mod : mods) {
        auto &indices = binded[mod.slot];
        deref(indices);
        using Ope = BindlessArrayUpdateCommand::Modification::Operation;
        if (mod.tex2d.op == Ope::EMPLACE) {
            indices = add_index(mod.tex2d.handle);
        }
    }
}
void BindlessArray::bind(vstd::span<const BindlessArrayUpdateCommand::BufferModification> mods) {
    LUISA_DEBUG_ASSERT(typed_binded.index() == 1 && _buffer_node);
    auto &binded = typed_binded.get<1>();
    std::lock_guard lck{mtx};
    if (mods.empty()) return;
    for (auto &&mod : mods) {
        auto &indices = binded[mod.slot];
        deref(indices);
        using Ope = BindlessArrayUpdateCommand::Modification::Operation;
        if (mod.buffer.op == Ope::EMPLACE) {
            indices = add_index(mod.buffer.handle);
        }
    }
}
auto BindlessArray::add_index(size_t ptr) -> Map::Index {
    auto ite = ptrMap.emplace(ptr, 0);
    ite.value()++;
    return ite;
}
void BindlessArray::bind(luisa::span<BindlessArrayUpdateCommand::Modification const> mods) {
    LUISA_DEBUG_ASSERT(typed_binded.index() == 0);
    auto &binded = typed_binded.get<0>();
    std::lock_guard lck{mtx};

    auto emplace_tex = [&]<bool isTex2D>(BindlessStruct &bind_grp, MapIndicies &indices, uint64_t handle, Texture const *tex, Sampler const &samp) {
        uint tex_idx;
        if constexpr (isTex2D) {
            return_value(indices.tex2D, 1, bind_grp.tex2D);
            tex_idx = device()->tex2d_heap_pool.alloc();
        } else {
            return_value(indices.tex2D, 2, bind_grp.tex2D);
            tex_idx = device()->tex3d_heap_pool.alloc();
        }
        auto smp_idx = luisa::to_underlying(samp.filter()) + luisa::to_underlying(samp.address()) * 4;
        // auto smpIdx = GlobalSamplers::GetIndex(samp);
        if constexpr (isTex2D) {
            indices.tex2D = add_index(handle);
            bind_grp.write_samp2d(tex_idx, smp_idx);
        } else {
            indices.tex3D = add_index(handle);
            bind_grp.write_samp3d(tex_idx, smp_idx);
        }
    };
    for (auto &&mod : mods) {
        auto &bind_grp = binded[mod.slot].first;
        auto &indices = binded[mod.slot].second;
        using Ope = BindlessArrayUpdateCommand::Modification::Operation;
        switch (mod.buffer.op) {
            case Ope::REMOVE:
                return_value(indices.buffer, 0, bind_grp.buffer);
                break;
            case Ope::EMPLACE: {
                return_value(indices.buffer, 0, bind_grp.buffer);
                auto buffer = reinterpret_cast<Buffer *>(mod.buffer.handle);
                BufferView v{buffer, mod.buffer.offset_bytes, buffer->byte_size() - mod.buffer.offset_bytes};
                auto new_idx = device()->buffer_heap_pool.alloc();
                bind_grp.buffer = new_idx;
                indices.buffer = add_index(mod.buffer.handle);
                break;
            }
            default: break;
        }
        switch (mod.tex2d.op) {
            case Ope::REMOVE:
                return_value(indices.tex2D, 1, bind_grp.tex2D);
                break;
            case Ope::EMPLACE:
                emplace_tex.operator()<true>(bind_grp, indices, mod.tex2d.handle, reinterpret_cast<Texture *>(mod.tex2d.handle), mod.tex2d.sampler);
                break;
            default: break;
        }
        switch (mod.tex3d.op) {
            case Ope::REMOVE:
                return_value(indices.tex3D, 2, bind_grp.tex3D);
                break;
            case Ope::EMPLACE:
                emplace_tex.operator()<false>(bind_grp, indices, mod.tex3d.handle, reinterpret_cast<Texture *>(mod.tex3d.handle), mod.tex3d.sampler);
                break;
            default: break;
        }
    }
}
void BindlessArray::copy_index(CommandBuffer *cmdbuffer) {
    if (offset_setted) return;
    offset_setted = true;
    auto dsc_buffer = cmdbuffer->states()->upload_alloc.allocate(4, 4);
    uint value = bdls_detail::get_alloc(*device(), _type).get_index(_buffer_node);
    reinterpret_cast<UploadBuffer const *>(dsc_buffer.buffer)->copy_from(&value, dsc_buffer.offset, sizeof(uint));
    VkBufferCopy2 buffer_copy{
        VK_STRUCTURE_TYPE_BUFFER_COPY_2,
        nullptr,
        dsc_buffer.offset,
        0,
        4};
    VkCopyBufferInfo2 copy_info2{
        VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
        nullptr,
        dsc_buffer.buffer->vk_buffer(),
        _indices_buffer.vk_buffer(),
        1,
        &buffer_copy};
    vkCmdCopyBuffer2(
        cmdbuffer->cmdbuffer(),
        &copy_info2);
}
void BindlessArray::update(
    CommandBuffer *cmdbuffer,
    luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
    luisa::vector<uint4> &cache,
    luisa::span<BindlessArrayUpdateCommand::BufferModification const> mods) {
    std::lock_guard lck{mtx};
    copy_index(cmdbuffer);
    using Ope = BindlessArrayUpdateCommand::Modification::Operation;
    for (auto &mod : mods) {
        if (mod.buffer.op == Ope::EMPLACE) {
            uint idx = device()->buffer_heap_pool.get_index(_buffer_node) + mod.slot;
            auto buffer = reinterpret_cast<Buffer *>(mod.buffer.handle);
            auto buffer_info = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
            BufferView v{buffer, mod.buffer.offset_bytes, buffer->byte_size() - mod.buffer.offset_bytes};
            *buffer_info = VkDescriptorBufferInfo{
                buffer->vk_buffer(),
                v.offset,
                v.size_bytes};
            write_desc_sets.emplace_back(VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                device()->bdls_buffer_set(),
                0,
                idx,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                buffer_info,
                nullptr});
        }
    }
    if (!write_desc_sets.empty()) {
        vkUpdateDescriptorSets(
            device()->logic_device(),
            write_desc_sets.size(),
            write_desc_sets.data(),
            0,
            nullptr);
        write_desc_sets.clear();
    }
}
void BindlessArray::update(
    CommandBuffer *cmdbuffer,
    luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
    luisa::vector<uint4> &cache,
    luisa::span<BindlessArrayUpdateCommand::Texture2DModification const> mods) {
    std::lock_guard lck{mtx};
    copy_index(cmdbuffer);
    using Ope = BindlessArrayUpdateCommand::Modification::Operation;
    for (auto &mod : mods) {
        if (mod.tex2d.op == Ope::EMPLACE) {
            auto idx = device()->tex2d_heap_pool.get_index(_buffer_node) + mod.slot;
            emplace_tex(cmdbuffer, write_desc_sets, device()->bdls_tex2d_set(), idx, reinterpret_cast<Texture *>(mod.tex2d.handle));
        }
    }
    if (!write_desc_sets.empty()) {
        vkUpdateDescriptorSets(
            device()->logic_device(),
            write_desc_sets.size(),
            write_desc_sets.data(),
            0,
            nullptr);
        write_desc_sets.clear();
    }
}
void BindlessArray::update(
    CommandBuffer *cmdbuffer,
    luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
    luisa::vector<uint4> &cache,
    luisa::span<BindlessArrayUpdateCommand::Texture3DModification const> mods) {
    std::lock_guard lck{mtx};
    copy_index(cmdbuffer);
    using Ope = BindlessArrayUpdateCommand::Modification::Operation;
    for (auto &mod : mods) {
        if (mod.tex3d.op == Ope::EMPLACE) {
            auto idx = device()->tex3d_heap_pool.get_index(_buffer_node) + mod.slot;
            emplace_tex(cmdbuffer, write_desc_sets, device()->bdls_tex3d_set(), idx, reinterpret_cast<Texture *>(mod.tex3d.handle));
        }
    }
    if (!write_desc_sets.empty()) {
        vkUpdateDescriptorSets(
            device()->logic_device(),
            write_desc_sets.size(),
            write_desc_sets.data(),
            0,
            nullptr);
        write_desc_sets.clear();
    }
}
void BindlessArray::emplace_tex(
    CommandBuffer *cmdbuffer,
    luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
    VkDescriptorSet tex_set,
    uint tex_idx,
    Texture const *tex) const {
    auto image_info = cmdbuffer->temp_desc->allocate_memory<VkDescriptorImageInfo>();
    auto &img_view = cmdbuffer->states()->img_views.emplace_back();
    VkImageViewCreateInfo img_view_create_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .flags = 0,
        .image = tex->vk_image(),
        .viewType = [&]() {
            switch (tex->dimension()) {
                case 1:
                    return VK_IMAGE_VIEW_TYPE_1D;
                case 2:
                    return VK_IMAGE_VIEW_TYPE_2D;
                case 3:
                    return VK_IMAGE_VIEW_TYPE_3D;
            }
        }(),
        .format = Texture::to_vk_format(tex->format()),
        .subresourceRange = VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = tex->mip(), .baseArrayLayer = 0, .layerCount = 1}};
    VK_CHECK_RESULT(vkCreateImageView(device()->logic_device(), &img_view_create_info, Device::alloc_callbacks(), &img_view));

    *image_info = VkDescriptorImageInfo{
        nullptr,
        img_view,
        cmdbuffer->resource_barrier->get_layout(tex, 0)};
    write_desc_sets.emplace_back(VkWriteDescriptorSet{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        tex_set,
        0,
        tex_idx,
        1,
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        image_info,
        nullptr,
        nullptr});
}
void BindlessArray::update(
    CommandBuffer *cmdbuffer,
    luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
    luisa::vector<uint4> &cache,
    luisa::span<BindlessArrayUpdateCommand::Modification const> mods) {
    LUISA_DEBUG_ASSERT(typed_binded.index() == 0);
    auto &binded = typed_binded.get<0>();
    std::lock_guard lck{mtx};
    auto dsc_buffer = cmdbuffer->states()->upload_alloc.allocate(16 * mods.size(), 16);
    auto shader = device()->set_bindless_kernel.Get(device());
    cache.clear();
    cache.reserve(mods.size());
    auto emplace_tex = [&]<bool isTex2D>(BindlessStruct &bind_grp, Texture const *tex) {
        VkDescriptorSet tex_set;
        uint tex_idx;
        if constexpr (isTex2D) {
            tex_set = device()->bdls_tex2d_set();
            tex_idx = bind_grp.tex2D & BindlessStruct::mask;
        } else {
            tex_set = device()->bdls_tex3d_set();
            tex_idx = bind_grp.tex3D & BindlessStruct::mask;
        }
        this->emplace_tex(
            cmdbuffer,
            write_desc_sets,
            tex_set,
            tex_idx,
            tex);
    };
    for (auto &mod : mods) {
        using Ope = BindlessArrayUpdateCommand::Modification::Operation;
        auto &bind_grp = binded[mod.slot].first;
        if (mod.buffer.op == Ope::EMPLACE) {
            auto buffer = reinterpret_cast<Buffer *>(mod.buffer.handle);
            auto buffer_info = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
            BufferView v{buffer, mod.buffer.offset_bytes, buffer->byte_size() - mod.buffer.offset_bytes};
            *buffer_info = VkDescriptorBufferInfo{
                buffer->vk_buffer(),
                v.offset,
                v.size_bytes};
            write_desc_sets.emplace_back(VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                device()->bdls_buffer_set(),
                0,
                bind_grp.buffer,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                buffer_info,
                nullptr});
        }
        if (mod.tex2d.op == Ope::EMPLACE) {
            emplace_tex.operator()<true>(bind_grp, reinterpret_cast<Texture *>(mod.tex2d.handle));
        }
        if (mod.tex3d.op == Ope::EMPLACE) {
            emplace_tex.operator()<true>(bind_grp, reinterpret_cast<Texture *>(mod.tex3d.handle));
        }
        auto &v = cache.emplace_back();
        v.x = mod.slot;
        std::memcpy(&v.y, &bind_grp, sizeof(BindlessStruct));
        static_assert(sizeof(BindlessStruct) == 12);
    }
    //
    VkDescriptorSet desc_set;
    VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = cmdbuffer->states()->_desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = shader->desc_set_layout().data()};
    VK_CHECK_RESULT(
        vkAllocateDescriptorSets(
            device()->logic_device(),
            &alloc_info,
            &desc_set));
    uint value = mods.size();
    vkCmdPushConstants(
        cmdbuffer->cmdbuffer(),
        shader->pipeline_layout(),
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        &value);
    VkDescriptorBufferInfo arg_buffer_info{
        dsc_buffer.buffer->vk_buffer(),
        dsc_buffer.offset,
        dsc_buffer.size_bytes};
    VkDescriptorBufferInfo buffer_info{
        _indices_buffer.vk_buffer(),
        0,
        _indices_buffer.byte_size()};
    static_cast<UploadBuffer const *>(dsc_buffer.buffer)->copy_from(cache.data(), dsc_buffer.offset, cache.size_bytes());

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
        cmdbuffer->cmdbuffer(),
        VK_PIPELINE_BIND_POINT_COMPUTE,
        shader->pipeline_layout(),
        0,
        1,
        &desc_set,
        0,
        nullptr);
    vkCmdBindPipeline(cmdbuffer->cmdbuffer(), VK_PIPELINE_BIND_POINT_COMPUTE, shader->pipeline());
    vkCmdDispatch(cmdbuffer->cmdbuffer(), (mods.size() + 255) / 256, 1, 1);
    if (!freeQueue.empty()) {
        cmdbuffer->states()->_callbacks.emplace_back([freeQueue = std::move(freeQueue), device = device()]() {
            for (auto &i : freeQueue) {
                switch (i.type) {
                    case 0:
                        device->buffer_heap_pool.dealloc(i.index);
                        break;
                    case 1:
                        device->tex2d_heap_pool.dealloc(i.index);
                        break;
                    case 2:
                        device->tex3d_heap_pool.dealloc(i.index);
                        break;
                }
            }
        });
    }
}
}// namespace lc::vk