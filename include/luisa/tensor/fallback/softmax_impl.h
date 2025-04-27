#pragma once
#include "i_tensor_expr_executor.h"
#include <luisa/core/stl/variant.h>
namespace luisa::compute {
struct SoftmaxImpl : ITensorExprExecutor {
    SoftmaxExpr *expr;
    struct LargeBatchShader {
        ShaderManager::ShaderDispatch sum;
        ShaderManager::ShaderDispatch final;
    };
    luisa::variant<
        LargeBatchShader,
        ShaderManager::ShaderDispatch>
        shaders;

    SoftmaxImpl(
        DeviceInterface *device,
        ShaderManager *shader_manager,
        SoftmaxExpr *expr);
    ~SoftmaxImpl();
    void execute(FallbackTensorCallback *callback, CommandList &cmdlist) const override;
};
};// namespace luisa::compute