#pragma once
#include <luisa/tensor/expression.h>
#include <luisa/tensor/tensor_interface.h>
#include <luisa/tensor/tensor_builder.h>
#include <luisa/runtime/device.h>
#include <luisa/vstl/meta_lib.h>
namespace luisa::compute {
class TensorInterface;
namespace detail {
template<typename Ret, typename... Args>
struct ObjType {
    template<template<typename...> typename Object, typename... ExtraTypes>
    using Type = typename Object<ExtraTypes..., Ret, Args...>;
};
template<typename T>
struct member_func_meta;

template<typename Ret, typename Class, typename... Args>
struct member_func_meta<Ret (Class::*)(Args...) const> : ObjType<Ret, Args...> {
    static constexpr bool is_const = true;
};
template<typename Ret, typename Class, typename... Args>
struct member_func_meta<Ret (Class::*)(Args...)> : ObjType<Ret, Args...> {
    static constexpr bool is_const = false;
};
template<typename Ret, typename Class, typename... Args>
struct member_func_meta<Ret (Class::*)(Args...) const noexcept> : ObjType<Ret, Args...> {
    static constexpr bool is_const = true;
};
template<typename Ret, typename Class, typename... Args>
struct member_func_meta<Ret (Class::*)(Args...) noexcept> : ObjType<Ret, Args...> {
    static constexpr bool is_const = false;
};
}// namespace detail

struct TensorDescriptor {
    luisa::fixed_vector<size_t, 4> _sizes;
    TensorElementType _type;
    TensorDescriptor(
        std::initializer_list<size_t> dimensions,
        TensorElementType type) noexcept : _type(type) {
        _sizes.resize_uninitialized(dimensions.size());
        std::memcpy(_sizes.data(), dimensions.begin(), _sizes.size_bytes());
    }
    TensorDescriptor(
        luisa::span<size_t const> dimensions,
        TensorElementType type) noexcept : _type(type) {
        _sizes.resize_uninitialized(dimensions.size());
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
    static_assert(luisa::always_false_v<Ret>, "Tensor kernel must not have return value.");
};

template<typename Lambda, typename... Args>
class TensorKernel<Lambda, void, Args...> {
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
        _lambda(detail::tensor_kernel_type<Args>::_forward(std::forward<typename detail::tensor_kernel_type<Args>::Type>(args))...);
        TensorBuilder::set_thd_local(nullptr);
        _kernel_ptr = _tensor_interface->compile_kernel(std::move(r));
    }
    template<typename... ExecArgs>
        requires (sizeof...(Args) == sizeof...(ExecArgs) && !(detail::AnyMap<luisa::compute::is_buffer_or_view, true>::template Run<std::remove_cvref_t<ExecArgs>...>()))
    void execute(CommandList &cmdlist, ExecArgs const &...args) noexcept {
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
        auto tensors = {to_desc(args)...};
        _tensor_interface->execute(cmdlist, _kernel_ptr, luisa::span<Argument::Buffer const>{tensors.begin(), tensors.size()});
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
    using KernelType = typename detail::member_func_meta<decltype(&std::remove_cvref_t<decltype(lambda)>::operator())>::Type<TensorKernel, Lambda>;
    return KernelType{tensor_interface, std::forward<Lambda>(lambda)};
}

}// namespace luisa::compute