#pragma once
#include <luisa/runtime/device.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/rhi/argument.h>
namespace luisa::compute {
class TensorBuilder;
class TensorInterface {
private:
    DeviceInterface *_device;

public:
    explicit TensorInterface(Device &device) noexcept : _device(device.impl()) {}
    virtual ~TensorInterface() noexcept = default;
    TensorInterface(TensorInterface const &) = delete;
    TensorInterface(TensorInterface &&) = delete;

    [[nodiscard]] virtual void *compile_kernel(luisa::unique_ptr<TensorBuilder> &&tensor_builder) noexcept = 0;
    virtual void destroy_kernel(void *kernel_ptr) noexcept = 0;

    [[nodiscard]] virtual void execute(
        CommandList& cmdlist,
        void *kernel_ptr,
        luisa::span<Argument::Buffer const> tensors) noexcept = 0;

    [[nodiscard]] DeviceInterface *device() const noexcept { return _device; }
    LC_TENSOR_API static luisa::unique_ptr<TensorInterface> create_fallback_backend(Device& device);
};
}// namespace luisa::compute