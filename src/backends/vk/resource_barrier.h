#pragma once
#include <vulkan/vulkan_core.h>
#include "buffer.h"
#include "texture.h"
#include <luisa/vstl/common.h>
namespace lc::vk {
class BindlessArray;
class Buffer;
class ResourceBarrier {
    struct Range {
        int64_t min;
        int64_t max;
        Range() {
            min = std::numeric_limits<int64_t>::min();
            max = std::numeric_limits<int64_t>::max();
        }
        Range(int64_t value) {
            min = value;
            max = value + 1;
        }
        Range(int64_t min, int64_t size)
            : min(min), max(size + min) {}
        bool collide(Range const &r) const {
            return min < r.max && r.min < max;
        }
        bool operator==(Range const &r) const {
            return min == r.min && max == r.max;
        }
        bool operator!=(Range const &r) const { return !operator==(r); }
    };
    struct BufferRange {
        // Range range;
        VkPipelineStageFlagBits2 before_stage{0};
        VkPipelineStageFlagBits2 after_stage{0};
        VkAccessFlagBits2 before_access{0};
        VkAccessFlagBits2 after_access{0};
        bool first_time{true};// used for backup
    };
    struct TextureRange {
        bool level_inited{false};
        bool level_require_update{false};
        bool first_time{true};// used for backup
        VkPipelineStageFlagBits2 before_stage{0};
        VkPipelineStageFlagBits2 after_stage{0};
        VkAccessFlagBits2 before_access{0};
        VkAccessFlagBits2 after_access{0};
        VkImageLayout before_layout{VK_IMAGE_LAYOUT_GENERAL};
        VkImageLayout after_layout{VK_IMAGE_LAYOUT_GENERAL};
    };
    struct ResourceStates {
        vstd::variant<
            BufferRange,
            vstd::vector<TextureRange>>
            layer_states;

        enum class Type : uint8_t {
            Buffer,
            Texture
        };
        size_t size;
        bool require_update{false};
        ResourceStates(Type type, size_t size);
    };
    struct BufferAfterRange {
        // Range range;
        VkPipelineStageFlagBits2 stage;
        VkAccessFlagBits2 access;
    };
    using ResourceView = vstd::variant<
        BufferView,
        TexView>;
    vstd::HashMap<Resource const *, ResourceStates> frame_states;
    vstd::vector<std::pair<Resource const *, ResourceStates *>> current_update_states;
    vstd::HashMap<Resource const *, size_t /* size */> write_state_map;
    vstd::vector<VkImageMemoryBarrier2> tex_barriers;
    vstd::vector<VkBufferMemoryBarrier2> buffer_barriers;
    void _update_state(Resource const *res_ptr, ResourceStates &states);
public:
    enum class Usage : uint {
        ComputeRead,
        ComputeAccelRead,
        ComputeUAV,
        CopySource,
        CopyDest,
        BuildAccel,
        CopyAccelSrc,
        CopyAccelDst,
        DepthRead,
        DepthWrite,
        IndirectArgs,
        VertexRead,
        IndexRead,
        RenderTarget,
        AccelInstanceBuffer,
        RasterRead,
        RasterAccelRead,
        RasterUAV,
    };
    enum class QueueType {
        Graphics,
        Compute,
        Copy
    };
    QueueType queue_type{QueueType::Graphics};
    uint queue_index{0};
    ResourceBarrier();
    void add_buffer(Buffer const *buffer, size_t offset, size_t size);
    ~ResourceBarrier();
    void record(
        ResourceView const &res,
        Usage usage);
    void record(
        ResourceView const &res,
        VkPipelineStageFlagBits2 stage,
        VkAccessFlagBits2 access,
        VkImageLayout layout);
    void update_states(
        VkCommandBuffer cmd_buffer);
    void restore_states(VkCommandBuffer cmd_buffer);
    void barrier_filter(VkBufferMemoryBarrier2 &barrier) const;
    void barrier_filter(VkImageMemoryBarrier2 &barrier) const;
    VkImageLayout get_layout(Resource const *res, uint level) const;
    void process_bindless(BindlessArray *bdls_arr, Usage dst_usage);
};
}// namespace lc::vk
