#pragma once
#include "buffer.h"
#include "vk_allocator.h"
namespace lc::vk {
class DefaultBuffer : public Buffer {
    AllocatedBuffer _res;

public:
    DefaultBuffer(Device *device, size_t size_bytes, bool used_as_accel = false, VkBufferUsageFlagBits extra_bit = (VkBufferUsageFlagBits)0);
    DefaultBuffer(DefaultBuffer&& rhs);
    ~DefaultBuffer();
    VkBuffer vk_buffer() const override { return _res.buffer; }
};
}// namespace lc::vk
