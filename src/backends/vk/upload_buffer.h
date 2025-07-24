#pragma once
#include "buffer.h"
#include "vk_allocator.h"
namespace lc::vk {
class UploadBuffer : public Buffer {
    AllocatedBuffer _res;
    void *_mapped_ptr{};

public:
    mutable BufferFlusher _flusher;
    UploadBuffer(Device *device, size_t size_bytes);
    ~UploadBuffer();
    void copy_from(void const *data, size_t offset, size_t size) const;
    VkBuffer vk_buffer() const override { return _res.buffer; }
    void *mapped_ptr() const { return _mapped_ptr; }
    bool flush_host() const override;
};
}// namespace lc::vk
