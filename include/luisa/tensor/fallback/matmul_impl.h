#pragma once
#include "i_tensor_expr_executor.h"
namespace luisa::compute {
struct MatMulImpl : ITensorExprExecutor {
    uint3 dispatch_size;
    size_t uniform_size;
    uint64_t shader_handle;
    DeviceInterface *device;
        GEMMExpr* expr;
    MatMulImpl(
        DeviceInterface *device,
        GEMMExpr* expr);
    ~MatMulImpl();
    void execute(FallbackTensorCallback *callback, CommandList &cmdlist) const override;
};
};// namespace luisa::compute