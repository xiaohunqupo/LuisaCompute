#pragma once
#include "resource.h"
#include <volk.h>
namespace lc::vk {
class Buffer : public Resource {
    size_t _byte_size;

public:
    Buffer(Device *device, size_t byte_size)
        : Resource{device},
          _byte_size{byte_size} {};
    Buffer(Buffer &&) = default;
    auto byte_size() const { return _byte_size; }
    virtual ~Buffer() = default;
    virtual VkBuffer vk_buffer() const = 0;
    Tag tag() const override { return Tag::Buffer; }
    uint64_t get_device_address() const;
};
class BufferView {
public:
    Buffer const *buffer;
    size_t offset;
    size_t size_bytes;
    BufferView() : buffer(nullptr), offset(0), size_bytes(0) {}
    BufferView(Buffer const *buffer) : buffer(buffer), offset(0), size_bytes(buffer->byte_size()) {
    }
    BufferView(
        Buffer const *buffer,
        size_t offset,
        size_t size_bytes)
        : buffer(buffer),
          offset(offset),
          size_bytes(size_bytes) {}
};

}// namespace lc::vk
