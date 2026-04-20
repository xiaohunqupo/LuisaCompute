#pragma once
#include "buffer.h"
#include "vk_allocator.h"
namespace lc::vk {
class ReadbackBuffer : public Buffer {
    AllocatedBuffer _res;
    void *_mapped_ptr{};

public:
    mutable BufferFlusher flusher;
    ReadbackBuffer(Device *device, size_t size_bytes);
    ~ReadbackBuffer();
    void copy_to(void *data, size_t offset, size_t size) const;
    VkBuffer vk_buffer() const override { return _res.buffer; }
    void *mapped_ptr() const { return _mapped_ptr; }
    bool flush_host() const override;
    void flush_range(size_t begin, size_t end) override;
};
}// namespace lc::vk
