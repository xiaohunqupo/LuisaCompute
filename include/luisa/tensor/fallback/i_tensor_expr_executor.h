#pragma once
#include <luisa/core/stl/unordered_map.h>
#include <luisa/runtime/command_list.h>
#include <luisa/tensor/expression.h>
#include <luisa/runtime/rhi/argument.h>
namespace luisa::compute {
struct FallbackTensorKenel;
struct FallbackTensorCallback {
    luisa::unordered_map<TensorData *, Argument::Buffer> *args;
    FallbackTensorKenel *kernel;
    Argument::Buffer get_tensor_buffer(TensorData *data);
};
struct ITensorExprExecutor {
    ~ITensorExprExecutor() = default;
    ITensorExprExecutor() = default;
    ITensorExprExecutor(ITensorExprExecutor const &) = delete;
    ITensorExprExecutor(ITensorExprExecutor &&) = delete;
    virtual void execute(FallbackTensorCallback *callback, CommandList &cmdlist) const = 0;
};
}// namespace luisa::compute