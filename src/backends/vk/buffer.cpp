#include "upload_buffer.h"
#include "readback_buffer.h"
#include "default_buffer.h"
#include "sparse_buffer.h"
#include "device.h"
#include "log.h"
namespace lc::vk {
UploadBuffer::UploadBuffer(Device *device, size_t size_bytes)
    : Buffer{device, size_bytes},
      _res{
          device->allocator()
              .allocate_buffer(
                  size_bytes,
                  (VkBufferUsageFlagBits)((uint)VK_BUFFER_USAGE_TRANSFER_SRC_BIT | (uint)VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                  AccessType::Upload)} {
}
UploadBuffer::~UploadBuffer() {
    device()->allocator().destroy_buffer(_res);
}
ReadbackBuffer::ReadbackBuffer(Device *device, size_t size_bytes)
    : Buffer{device, size_bytes},
      _res{
          device->allocator()
              .allocate_buffer(
                  size_bytes,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  AccessType::ReadBack)} {
}
ReadbackBuffer::~ReadbackBuffer() {
    device()->allocator().destroy_buffer(_res);
}
void UploadBuffer::copy_from(void const *data, size_t offset, size_t size) const {
    void *mapped_ptr;
    VK_CHECK_RESULT(vmaMapMemory(
        device()->allocator().allocator(),
        _res.allocation,
        &mapped_ptr));
    memcpy(reinterpret_cast<std::byte *>(mapped_ptr) + offset, data, size);
    vmaFlushAllocation(
        device()->allocator().allocator(),
        _res.allocation,
        offset, size);
    vmaUnmapMemory(
        device()->allocator().allocator(),
        _res.allocation);
}
void ReadbackBuffer::copy_to(void *data, size_t offset, size_t size) const {
    void *mapped_ptr;
    VK_CHECK_RESULT(vmaMapMemory(
        device()->allocator().allocator(),
        _res.allocation,
        &mapped_ptr));
    memcpy(data, reinterpret_cast<std::byte *>(mapped_ptr) + offset, size);
    vmaFlushAllocation(
        device()->allocator().allocator(),
        _res.allocation,
        offset, size);
    vmaUnmapMemory(
        device()->allocator().allocator(),
        _res.allocation);
}
DefaultBuffer::DefaultBuffer(Device *device, size_t size_bytes, bool used_as_accel, VkBufferUsageFlagBits extra_bit)
    : Buffer{device, size_bytes},
      _res{
          device->allocator()
              .allocate_buffer(
                  size_bytes,
                  static_cast<VkBufferUsageFlagBits>(
                      extra_bit |
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                      VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                      VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT |
                      (used_as_accel ? (VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) :
                                       0)),
                  AccessType::None)} {
}
DefaultBuffer::~DefaultBuffer() {
    if (_res.buffer)
        device()->allocator().destroy_buffer(_res);
}
uint64_t Buffer::get_device_address() const {
    VkBufferDeviceAddressInfoKHR buffer_device_address_info{};
    buffer_device_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    buffer_device_address_info.buffer = vk_buffer();
    return vkGetBufferDeviceAddress(device()->logic_device(), &buffer_device_address_info);
}
DefaultBuffer::DefaultBuffer(DefaultBuffer &&rhs)
    : Buffer(std::move(rhs)) {
    _res = rhs._res;
    rhs._res.buffer = nullptr;
}
SparseBuffer::SparseBuffer(Device *device, size_t size_bytes, bool used_as_accel, VkBufferUsageFlagBits extra_bit)
    : Buffer(device, size_bytes) {
    VkBufferCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT,
        .size = size_bytes,
        .usage = static_cast<VkBufferUsageFlags>(
            extra_bit |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
            VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT |
            (used_as_accel ? (VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) :
                             0)),
        .queueFamilyIndexCount = 0};
    VK_CHECK_RESULT(vkCreateBuffer(
        device->logic_device(),
        &create_info,
        Device::alloc_callbacks(),
        &_buffer));
}
SparseBuffer::SparseBuffer(SparseBuffer &&rhs)
    : Buffer(std::move(rhs)) {
    _buffer = rhs._buffer;
    rhs._buffer = nullptr;
}
SparseBuffer::~SparseBuffer() {
    if (_buffer) {
        vkDestroyBuffer(device()->logic_device(), _buffer, Device::alloc_callbacks());
    }
}
}// namespace lc::vk
