#pragma once
#include <luisa/backends/ext/pinned_memory_ext.hpp>

namespace lc::vk {
class Device;
using namespace luisa;
using namespace luisa::compute;
class VkPinnedMemoryExt : public PinnedMemoryExt {
    Device *_device;
public:
    [[nodiscard]] BufferCreationInfo _pin_host_memory(
        const Type *elem_type, size_t elem_count,
        void *host_ptr, const PinnedMemoryOption &option) noexcept override;

    [[nodiscard]] BufferCreationInfo _allocate_pinned_memory(
        const Type *elem_type, size_t elem_count,
        const PinnedMemoryOption &option) noexcept override;
public:
    explicit VkPinnedMemoryExt(Device *device) : _device(device) {}
    [[nodiscard]] DeviceInterface *device() const noexcept override;
};
}// namespace kc::vk