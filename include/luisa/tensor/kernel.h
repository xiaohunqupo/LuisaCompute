#pragma once
#include <luisa/tensor/expression.h>
#include <luisa/tensor/tensor_interface.h>
#include <luisa/tensor/tensor_builder.h>
#include <luisa/runtime/device.h>
#include <luisa/vstl/meta_lib.h>
namespace luisa::compute {
class TensorInterface;
namespace detail {
template<template<typename...> typename Object, typename ExtraTypes, typename Ret, typename... Args>
struct ObjType {
    using Type = Object<ExtraTypes, Ret, Args...>;
};
template<template<typename...> typename Object, typename ExtraTypes, typename T>
struct member_func_meta;

template<template<typename...> typename Object, typename ExtraTypes, typename Ret, typename Class, typename... Args>
struct member_func_meta<Object, ExtraTypes, Ret (Class::*)(Args...) const> : ObjType<Object, ExtraTypes, Ret, Args...> {
    static constexpr bool is_const = true;
};
template<template<typename...> typename Object, typename ExtraTypes, typename Ret, typename Class, typename... Args>
struct member_func_meta<Object, ExtraTypes, Ret (Class::*)(Args...)> : ObjType<Object, ExtraTypes, Ret, Args...> {
    static constexpr bool is_const = false;
};
template<template<typename...> typename Object, typename ExtraTypes, typename Ret, typename Class, typename... Args>
struct member_func_meta<Object, ExtraTypes, Ret (Class::*)(Args...) const noexcept> : ObjType<Object, ExtraTypes, Ret, Args...> {
    static constexpr bool is_const = true;
};
template<template<typename...> typename Object, typename ExtraTypes, typename Ret, typename Class, typename... Args>
struct member_func_meta<Object, ExtraTypes, Ret (Class::*)(Args...) noexcept> : ObjType<Object, ExtraTypes, Ret, Args...> {
    static constexpr bool is_const = false;
};
}// namespace detail

struct TensorDescriptor {
    luisa::fixed_vector<uint64_t, 4> _sizes;
    TensorElementType _type;
    TensorDescriptor(
        std::initializer_list<uint64_t> dimensions,
        TensorElementType type) noexcept : _type(type) {
        luisa::vector_resize(_sizes, dimensions.size());
        std::memcpy(_sizes.data(), dimensions.begin(), _sizes.size_bytes());
    }
    TensorDescriptor(
        luisa::span<uint64_t const> dimensions,
        TensorElementType type) noexcept : _type(type) {
        luisa::vector_resize(_sizes, dimensions.size());
        std::memcpy(_sizes.data(), dimensions.data(), _sizes.size_bytes());
    }
};

namespace detail {
template<template<typename...> class map, bool reverse, typename... Tar>
struct AnyMap {
    template<typename T, typename... Args>
    static constexpr bool Run() {
        if constexpr ((map<T, Tar...>::value) ^ reverse) {
            return true;
        } else if constexpr (sizeof...(Args) == 0) {
            return false;
        } else {
            return Run<Args...>();
        }
    }
};

template<typename T>
struct tensor_kernel_type {
    using Type = T;
    static T &&_forward(T &&v) noexcept {
        return std::forward<T>(v);
    }
};

template<>
struct tensor_kernel_type<Tensor> {
    using Type = TensorDescriptor;
    static Tensor _forward(TensorDescriptor &&v) noexcept {
        auto tensor_builder = TensorBuilder::get_thd_local();
        auto tensor_data = tensor_builder->allocate_tensor(v._sizes, v._type);
        tensor_builder->push_argument(tensor_data);
        return Tensor{tensor_data};
    }
};
}// namespace detail

template<typename Lambda, typename Ret, typename... Args>
struct TensorKernel {
    Lambda _lambda;
    TensorInterface *_tensor_interface;
    void *_kernel_ptr{nullptr};
public:
    explicit TensorKernel(
        TensorInterface *tensor_interface,
        Lambda &&lambda) noexcept
        : _lambda(std::forward<Lambda>(lambda)),
          _tensor_interface(tensor_interface) {}

    void compile(
        typename detail::tensor_kernel_type<Args>::Type... args) noexcept {
        if (_kernel_ptr) {
            _tensor_interface->destroy_kernel(_kernel_ptr);
            _kernel_ptr = nullptr;
        }
        auto r = luisa::make_unique<TensorBuilder>();
        TensorBuilder::set_thd_local(r.get());
        auto tensors = std::tuple<std::remove_cvref_t<Args>...>{detail::tensor_kernel_type<Args>::_forward(std::forward<typename detail::tensor_kernel_type<Args>::Type>(args))...};
        using RetType = decltype(std::apply(_lambda, std::move(tensors)));
        if constexpr (std::is_same_v<RetType, void>) {
            std::apply(_lambda, std::move(tensors));
        } else if constexpr (std::is_same_v<RetType, Tensor>) {
            auto ret_tensor = std::apply(_lambda, std::move(tensors));
            ;
            r->push_output(ret_tensor.data());
        }
        TensorBuilder::set_thd_local(nullptr);
        _kernel_ptr = _tensor_interface->compile_kernel(std::move(r));
    }
    template<typename... ExecArgs>
        requires(sizeof...(Args) == sizeof...(ExecArgs) && (sizeof...(Args) == 0 || !(detail::AnyMap<luisa::compute::is_buffer_or_view, true>::template Run<std::remove_cvref_t<ExecArgs>...>())))
    luisa::vector<BufferView<float4>> execute(CommandList &cmdlist, ExecArgs const &...args) noexcept {
        auto to_desc = []<typename T>(T const &arg) -> Argument::Buffer {
            if constexpr (is_buffer_view_v<T>) {
                return Argument::Buffer{
                    arg.handle(),
                    arg.offset_bytes(),
                    arg.size_bytes()};
            } else {
                return Argument::Buffer{
                    arg.handle(),
                    0ull,
                    arg.size_bytes()};
            }
        };
        luisa::vector<BufferCreationInfo> buffer_handles;
        luisa::vector<BufferView<float4>> buffers;
        if constexpr (sizeof...(Args) > 0) {
            auto tensors = {to_desc(args)...};
            _tensor_interface->execute(cmdlist, _kernel_ptr, luisa::span<Argument::Buffer const>{tensors.begin(), tensors.size()}, buffer_handles);
        } else {
            _tensor_interface->execute(cmdlist, _kernel_ptr, luisa::span<Argument::Buffer const>{}, buffer_handles);
        }
        buffers.reserve(buffer_handles.size());
        for (auto &i : buffer_handles) {
            buffers.emplace_back(
                i.native_handle,
                i.handle,
                sizeof(float4),
                0,
                i.total_size_bytes / sizeof(float4),
                i.total_size_bytes);
        }
        return buffers;
    }
    // template<typename... Args>
    //     requires (detail::AnyMap<luisa::compute::is_buffer_or_view, true>::template Run<std::remove_cvref_t<Args>...>())
    // CommandList execute(Args const &...args) noexcept {
    //     CommandList cmdlist;
    //     execute(cmdlist, args...);
    //     return cmdlist;
    // }

    ~TensorKernel() noexcept {
        if (_kernel_ptr) {
            _tensor_interface->destroy_kernel(_kernel_ptr);
        }
    }
};

template<typename Lambda>
static auto load_tensor(
    TensorInterface *tensor_interface,
    Lambda &&lambda) noexcept {
    using FuncType = decltype(&std::remove_cvref_t<decltype(lambda)>::operator());
    using MemberFuncMeta = typename detail::member_func_meta<TensorKernel, Lambda, FuncType>;
    using KernelType = MemberFuncMeta::Type;
    return KernelType{tensor_interface, std::forward<Lambda>(lambda)};
}

}// namespace luisa::compute