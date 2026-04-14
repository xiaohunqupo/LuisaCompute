#pragma once
#include <luisa/ast/function.h>
#include <luisa/ast/function_builder.h>
#include <luisa/dsl/func.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>

namespace luisa::compute {

class LUISA_RUNTIME_API BuiltinKernel {
    Device *_device{nullptr};
    
public:
    explicit BuiltinKernel(Device *device) noexcept;
    
    // Static methods return shared_ptr so the builder stays alive
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_buffer() noexcept;
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_image_uint() noexcept;
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_image_int() noexcept;
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_image_float() noexcept;
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_volume_uint() noexcept;
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_volume_int() noexcept;
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_volume_float() noexcept;
    
    template<size_t N, typename... Args>
    [[nodiscard]] auto compile(luisa::shared_ptr<const detail::FunctionBuilder> builder) noexcept -> Shader<N, Args...> {
        Kernel<N, Args...> kernel{builder};
        return _device->compile(kernel);
    }
    
    // Cache for each fill operation
    Shader<1, Buffer<uint>, uint> _fill_buffer_uint;
    Shader<1, Buffer<int>, int> _fill_buffer_int;
    Shader<1, Buffer<float>, float> _fill_buffer_float;
    Shader<2, Image<uint>, uint> _fill_image_uint;
    Shader<2, Image<int>, int> _fill_image_int;
    Shader<2, Image<float>, float> _fill_image_float;
    Shader<3, Volume<uint>, uint> _fill_volume_uint;
    Shader<3, Volume<int>, int> _fill_volume_int;
    Shader<3, Volume<float>, float> _fill_volume_float;
};



}
