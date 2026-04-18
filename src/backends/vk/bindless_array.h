#pragma once
#include "default_buffer.h"
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/core/first_fit.h>

namespace lc::vk {
using namespace luisa::compute;
class CommandBuffer;
class Texture;
class ResourceBarrier;
class BindlessArray : public Resource {
public:
    using Map = vstd::HashMap<size_t, size_t>;
    using MapIndex = typename Map::Index;
    struct BindlessStruct {
        static constexpr auto kInvalidPos = std::numeric_limits<uint>::max();
        static constexpr auto kMask = (1u << 28u) - 1;
        uint buffer = kInvalidPos;
        uint tex_2d = kInvalidPos;
        uint tex_3d = kInvalidPos;
        void write_samp2d(uint tex, uint s) {
            tex_2d = tex | (s << 28);
        }
        void write_samp3d(uint tex, uint s) {
            tex_3d = tex | (s << 28);
        }
    };
    struct MapIndicies {
        MapIndex buffer;
        MapIndex tex_2d;
        MapIndex tex_3d;
    };
private:
    struct FreeValue {
        uint type : 2;
        uint index : 30;
    };
    DefaultBuffer _indices_buffer;
    BindlessSlotType _type;
    luisa::FirstFit::Node *_buffer_node = nullptr;
    bool _offset_setted = false;
    vstd::variant<
        vstd::vector<std::pair<BindlessStruct, MapIndicies>>,
        vstd::vector<MapIndex>>
        _typed_binded;
    Map _ptr_map;
    mutable vstd::vector<FreeValue> _free_queue;
    void _return_value(MapIndex &index, uint type, uint &origin_value);
    void _emplace_tex(
        VkImageView &img_view,
        CommandBuffer *cmdbuffer,
        luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
        VkDescriptorSet tex_set,
        uint tex_idx,
        Texture const *tex) const;

public:
    auto &indices_buffer() { return _indices_buffer; }
    auto const &indices_buffer() const { return _indices_buffer; }
    mutable std::mutex mtx;
    BindlessArray(Device *device, BindlessSlotType type, size_t size);
    void pre_update(ResourceBarrier *barrier);
    bool is_ptr_in_bindless(size_t ptr) const {
        return _ptr_map.find(ptr);
    }
    void deref(Map::Index &index);
    Map::Index add_index(size_t ptr);
    void bind(luisa::span<BindlessArrayUpdateCommand::Modification const> mods);
    void bind(vstd::span<const BindlessArrayUpdateCommand::BufferModification> mods);
    void bind(vstd::span<const BindlessArrayUpdateCommand::Texture2DModification> mods);
    void copy_index(CommandBuffer *cmdbuffer);
    void update(
        CommandBuffer *cmdbuffer,
        luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
        luisa::vector<uint4> &cache,
        luisa::span<BindlessArrayUpdateCommand::BufferModification const> mods);
    void update(
        CommandBuffer *cmdbuffer,
        luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
        luisa::vector<uint4> &cache,
        luisa::span<BindlessArrayUpdateCommand::Texture2DModification const> mods);
    void update(
        CommandBuffer *cmdbuffer,
        luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
        luisa::vector<uint4> &cache,
        luisa::span<BindlessArrayUpdateCommand::Texture3DModification const> mods);
    void update(
        CommandBuffer *cmdbuffer,
        luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
        luisa::vector<uint4> &cache,
        luisa::span<BindlessArrayUpdateCommand::Modification const> mods);
    ~BindlessArray();
};
}// namespace lc::vk