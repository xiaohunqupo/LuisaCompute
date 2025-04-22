#pragma once
#include <luisa/tensor/tensor_interface.h>
#include <luisa/tensor/pass/expr_topo.h>
#include "i_tensor_expr_executor.h"
namespace luisa::compute {
struct FallbackTensorKenel {
    DeviceInterface* device;
    luisa::unique_ptr<TensorBuilder> tensor_builder;
    ExprTopo expr_topo;
    luisa::vector<luisa::unique_ptr<ITensorExprExecutor>> executors;
    luisa::vector<size_t> tensor_sub_nodes;
    BufferCreationInfo tensor_buffer;
    FallbackTensorKenel(luisa::unique_ptr<TensorBuilder> &&tensor_builder);
    ~FallbackTensorKenel();
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