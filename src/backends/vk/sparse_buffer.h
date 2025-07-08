#pragma once
#include "buffer.h"
#include "vk_allocator.h"
namespace lc::vk {
class SparseBuffer : public Buffer {
    VkBuffer _buffer{};

public:
    SparseBuffer(Device *device, size_t size_bytes, bool used_as_accel = false, VkBufferUsageFlagBits extra_bit = (VkBufferUsageFlagBits)0);
    SparseBuffer(SparseBuffer &&rhs);
    ~SparseBuffer();
    VkBuffer vk_buffer() const override { return _buffer; }
};
}// namespace lc::vk