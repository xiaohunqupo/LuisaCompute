#pragma once
#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/pool.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/tensor/fused_activation.h>
namespace luisa::compute {

enum struct TensorElementType : uint8_t {
    Float16,
    Float32,
    Float64,
};
template<typename T>
constexpr TensorElementType get_tensor_elem_type() {
    if constexpr (std::is_same_v<T, half>) {
        return TensorElementType::Float16;
    } else if constexpr (std::is_same_v<T, float>) {
        return TensorElementType::Float32;
    } else if constexpr (std::is_same_v<T, double>) {
        return TensorElementType::Float64;
    } else {
        static_assert(luisa::always_false_v<T>, "Bad type.");
    }
}
template<typename T>
concept valid_tensor_elem_type = requires() {
    get_tensor_elem_type<T>();
};
constexpr uint64_t tensor_element_size(TensorElementType type) {
    switch (type) {
        case TensorElementType::Float16:
            return 2;
        case TensorElementType::Float32:
            return 4;
        case TensorElementType::Float64:
            return 8;
        default:
            return 0;
    }
}
constexpr uint64_t tensor_element_align(TensorElementType type) {
    switch (type) {
        case TensorElementType::Float16:
            return 2;
        case TensorElementType::Float32:
            return 4;
        case TensorElementType::Float64:
            return 8;
        default:
            return 0;
    }
}
class LC_TENSOR_API TensorData {
    luisa::span<uint64_t const> _sizes;
    TensorElementType _type;
    uint64_t _idx;
    uint64_t _size_bytes;

public:
    TensorData(luisa::span<uint64_t const> sizes,
               TensorElementType element_type,
               uint64_t uid) noexcept;
    TensorData(TensorData &&rhs) noexcept;

    TensorData(TensorData const &rhs) = delete;

    [[nodiscard]] uint64_t idx() const noexcept {
        return _idx;
    }
    [[nodiscard]] uint64_t get_size(uint dimension) const noexcept {
        if (dimension >= _sizes.size()) return 1;
        return _sizes[dimension];
    }
    [[nodiscard]] uint64_t dimension() const noexcept {
        return _sizes.size();
    }
    [[nodiscard]] uint64_t size_bytes() const noexcept {
        return _size_bytes;
    }
    [[nodiscard]] TensorElementType element_type() const noexcept {
        return _type;
    }
};

class TensorBuilder;

class LC_TENSOR_API Tensor {
    friend class TensorBuilder;

    TensorData *_data;
    bool _contained;
    [[nodiscard]] void _create(TensorElementType element_type, luisa::span<const uint64_t> sizes, Argument::Buffer buffer) noexcept;

public:
    explicit Tensor(TensorData *data,
                    bool contained = true) noexcept;
    Tensor(Tensor &&rhs) noexcept;
    ~Tensor() noexcept;
    Tensor &operator=(Tensor &&rhs) noexcept {
        if (&rhs == this) [[unlikely]]
            return *this;
        std::destroy_at(this);
        new (this) Tensor(std::move(rhs));
    }
    template<typename T>
        requires(luisa::compute::is_buffer_or_view_v<T> && valid_tensor_elem_type<luisa::compute::buffer_element_t<T>>)
    Tensor(T const &t, luisa::span<const uint64_t> sizes) noexcept {
        Argument::Buffer bf{
            .handle = t.handle(),
            .size = t.size_bytes()};
        if constexpr (luisa::compute::is_buffer_v<T>) {
            bf.offset = 0;
        } else {
            bf.offset = t.offset_bytes();
        }
        _create(get_tensor_elem_type<luisa::compute::buffer_element_t<T>>(), sizes, bf);
    }
    template<typename T>
        requires(luisa::compute::is_buffer_or_view_v<T> && valid_tensor_elem_type<luisa::compute::buffer_element_t<T>>)
    Tensor(T const &t, std::initializer_list<const uint64_t> sizes) noexcept
        : Tensor(t, luisa::span{sizes.begin(), sizes.size()}) {
    }
    template<typename T>
        requires(luisa::compute::is_buffer_or_view_v<T> && valid_tensor_elem_type<luisa::compute::buffer_element_t<T>>)
    Tensor(T const &t) noexcept {
        Argument::Buffer bf{
            .handle = t.handle(),
            .size = t.size_bytes()};
        if constexpr (luisa::compute::is_buffer_v<T>) {
            bf.offset = 0;
        } else {
            bf.offset = t.offset_bytes();
        }
        uint64_t size = t.size_bytes();
        _create(get_tensor_elem_type<luisa::compute::buffer_element_t<T>>(), {&size, 1});
    }
    [[nodiscard]] auto data() const noexcept { return _data; }
    void dispose() noexcept;

    [[nodiscard]] static Tensor matmul(
        Tensor const &lhs,
        Tensor const &rhs,
        FusedActivation const &activation) noexcept;
    [[nodiscard]] static void init_tensor(
        Tensor const &input,
        uint64_t value) noexcept;
    [[nodiscard]] static void init_tensor(
        Tensor const &input,
        void *ptr,
        void (*disposer)(void *)) noexcept;

    [[nodiscard]] static Tensor conv_1d(
        Tensor const &input,
        Tensor const &weight,
        FusedActivation const &activation,
        TensorElementType out_type,
        uint filter_radius,
        uint dilation,
        uint start_padding = std::numeric_limits<uint>::max(),
        uint end_padding = std::numeric_limits<uint>::max()) noexcept;

    [[nodiscard]] static Tensor conv_2d(
        Tensor const &input,
        Tensor const &weight,
        FusedActivation const &activation,
        TensorElementType out_type,
        uint2 filter_radius,
        uint2 dilation,
        uint2 start_padding = uint2(std::numeric_limits<uint>::max()),
        uint2 end_padding = uint2(std::numeric_limits<uint>::max())) noexcept;

    [[nodiscard]] static Tensor conv_3d(
        Tensor const &input,
        Tensor const &weight,
        FusedActivation const &activation,
        TensorElementType out_type,
        uint3 filter_radius,
        uint3 dilation,
        uint3 start_padding = uint3(std::numeric_limits<uint>::max()),
        uint3 end_padding = uint3(std::numeric_limits<uint>::max())) noexcept;
};

// class DTensor {
//     Device &device;
//     bool _requires_grad = false;
//     bool _reserve_memory = false;
//     bool _dirty = false;
//     std::array<uint64_t, 3> _shape;
//     std::array<uint64_t, 3> _stride;
//     luisa::optional<Buffer<T>> _buffer;
//     luisa::optional<Var<T>> _var;
// public:
//     using shape_type = std::array<uint64_t, 3>;
//     using value_type = T;

// private:
//     static uint64_t compute_size(shape_type s) {
//         uint64_t size = 1;
//         for (auto i : s) {
//             size *= i;
//         }
//         return size;
//     }
// public:

//     explicit DTensor(Device &device) : device(device) {}

//     DTensor(Device &device, Buffer<T> buffer, shape_type shape) : device(device), _buffer(buffer), _shape(shape) {
//     }
//     static DTensor zeros(Device &device, shape_type shape) noexcept {
//         auto size = compute_size(shape);
//         auto tensor = Tensor{device, device.create_buffer<T>(size), shape};
//     }
//     void fill(const T &value) noexcept {
//     }
//     [[nodiscard]] luisa::optional<Buffer<T> &> buffer() noexcept;
//     [[nodiscard]] Tensor &requires_grad(bool requires_grad) noexcept {
//         _requires_grad = requires_grad;
//         return *this;
//     }
//     [[nodiscard]] Tensor &reserve_memory(bool reserve_memory) noexcept {
//         _reserve_memory = reserve_memory;
//         return *this;
//     }
//     // inplace operations
//     void scatter_(const Tensor<uint, Dim> &index, const Tensor<T, Dim> &src) noexcept {
//     }
//     [[nodiscard]] DTensor<T> gather(const DTensor<uint> &index) const noexcept {
//         // TODO: implement
//         return DTensor<T>{device};
//     }

//     // template<uint64_t... Is>
//     // [[nodiscard]] Tensor<T, Dim + sizeof...(Is)> repeat(Is...) {
//     //     // TODO: implement
//     //     return Tensor<T, Dim>{device};
//     // }
// };

// template<class R, uint64_t Dim, class... Ts>
// Tensor<R, Dim> map(const Tensor<Ts, Dim> &... ts) noexcept {
//     // TODO: implement
// }

}// namespace luisa::compute