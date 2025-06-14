#pragma once
#include <vulkan/vulkan.h>
#include "buffer.h"
#include "texture.h"
#include <luisa/vstl/common.h>
namespace lc::vk {
class Buffer;
class ResourceBarrier {
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
        VkPipelineStageFlagBits2 before_stage;
        VkPipelineStageFlagBits2 after_stage;
        VkAccessFlagBits2 before_access;
        VkAccessFlagBits2 after_access;
        bool first_time{true};// used for backup
    };
    struct TextureRange {
        bool level_inited{false};
        bool level_require_update;
        bool first_time{true};// used for backup
        VkPipelineStageFlagBits2 before_stage;
        VkPipelineStageFlagBits2 after_stage;
        VkAccessFlagBits2 before_access;
        VkAccessFlagBits2 after_access;
        VkImageLayout before_layout;
        VkImageLayout after_layout;
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
        ResourceStates(Type type, size_t size, Resource const *res);
    };
    using ResourceView = vstd::variant<
        BufferView,
        TexView>;
    vstd::HashMap<Resource const *, ResourceStates> frame_states;
    vstd::vector<std::pair<Resource const *, ResourceStates *>> current_update_states;
    vstd::HashMap<Resource const *, size_t /* size */> write_state_map;

public:
    ResourceBarrier();
    void add_buffer(Buffer const *buffer, size_t offset, size_t size);
    ~ResourceBarrier();
    void Record(
        ResourceView const &res,
        Usage usage);
    void Record(
        ResourceView const &res,
        VkPipelineStageFlagBits2 stage,
        VkAccessFlagBits2 access,
        VkImageLayout layout);
};
}// namespace lc::vk
