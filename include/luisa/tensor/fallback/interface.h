#pragma once
#include <luisa/tensor/tensor_interface.h>
namespace luisa::compute {
struct FallbackTensorKenel {
    luisa::unique_ptr<TensorBuilder> tensor_builder;
    FallbackTensorKenel(luisa::unique_ptr<TensorBuilder> &&tensor_builder);
    void check(luisa::span<Argument::Buffer const> tensors);
};

class FallbackTensorInterface : public TensorInterface {
public:
    FallbackTensorInterface(Device &device) noexcept : TensorInterface(device) {}
    void *compile_kernel(luisa::unique_ptr<TensorBuilder> &&tensor_builder) noexcept override;
    void destroy_kernel(void *kernel_ptr) noexcept override;
    void execute(
        CommandList &cmdlist,
        void *kernel_ptr,
        luisa::span<Argument::Buffer const> tensors) noexcept override;
};

}// namespace luisa::compute