#include <luisa/tensor/fallback/set_value_impl.h>
#include <luisa/dsl/sugar.h>
namespace luisa::compute {
namespace set_value_detail {
Kernel1D<Buffer<uint2>, uint2> set_value_kernel() {
    return [](BufferVar<uint2> buffer, UInt2 value) {
        set_block_size(128, 1, 1);
        buffer.write(dispatch_id().x, value);
    };
}
}// namespace set_value_detail
SetValueImpl::SetValueImpl(
    DeviceInterface *device,
    ShaderManager *shader_manager,
    SetValueExpr *expr)
    : expr(expr) {
    shader = shader_manager->add_shader(TensorExpr::Tag::ESetValueExpr, vstd::MD5{vstd::MD5::MD5Data{0ull, 0ull}}, [&]() {
        auto kernel = set_value_detail::set_value_kernel();
        return ShaderManager::ShaderDispatch{
            .shader_handle = device->create_shader(ShaderOption{}, Function{kernel.function().get()}).handle,
            .desired_dispatch_size = uint3(0),
            .uniform_size = ShaderDispatchCmdEncoder::compute_uniform_size(kernel.function()->unbound_arguments())};
    });
}
SetValueImpl::~SetValueImpl() {
}
void SetValueImpl::execute(FallbackTensorCallback *callback, CommandList &cmdlist) const {
    auto buffer = callback->get_tensor_buffer(expr->tensor_data);
    if (expr->value.index() == 1) {
        ComputeDispatchCmdEncoder encoder{
            shader.shader_handle,
            2,
            shader.uniform_size};
        encoder.encode_buffer(buffer.handle, buffer.offset, buffer.size);
        encoder.encode_uniform(&luisa::get<1>(expr->value), sizeof(uint2), alignof(uint2));
        encoder.set_dispatch_size(uint3(std::max<uint>(expr->tensor_data->size_bytes() / sizeof(uint2), 1), 1, 1));
        cmdlist << std::move(encoder).build();
    } else {
        cmdlist << luisa::make_unique<BufferUploadCommand>(
            buffer.handle,
            buffer.offset,
            buffer.size,
            luisa::get<0>(expr->value).ptr);
    }
}
}// namespace luisa::compute