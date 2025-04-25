#pragma once
#include "i_tensor_expr_executor.h"
namespace luisa::compute {
struct SetValueImpl : ITensorExprExecutor {
    SetValueExpr *expr;
    ShaderManager::ShaderDispatch shader;
    SetValueImpl(
        DeviceInterface *device,
        ShaderManager* shader_manager,
        SetValueExpr *expr);
    ~SetValueImpl();
    void execute(FallbackTensorCallback *callback, CommandList &cmdlist) const override;
};
}// namespace luisa::compute