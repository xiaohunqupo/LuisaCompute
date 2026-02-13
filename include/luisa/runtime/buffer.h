#pragma once

#include <luisa/core/mathematics.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace lc::validation {
class Stream;
}// namespace lc::validation

namespace luisa::compute {

namespace detail {

template<typename BufferOrView>
class BufferExprProxy;

LUISA_RUNTIME_API void error_buffer_copy_sizes_mismatch(size_t src, size_t dst) noexcept;
LUISA_RUNTIME_API void error_buffer_reinterpret_size_too_small(size_t size, size_t dst) noexcept;
LUISA_RUNTIME_API void error_buffer_subview_overflow(size_t offset, size_t ele_size, size_t size) noexcept;
LUISA_RUNTIME_API void error_buffer_invalid_alignment(size_t offset, size_t dst) noexcept;
LUISA_RUNTIME_API void error_buffer_size_is_zero() noexcept;

template<typename T>
struct is_buffer_impl : std::false_type {};

template<typename T>
struct is_buffer_view_impl : std::false_type {};

template<typename T>
struct buffer_element_impl {
    using type = T;
};

}// namespace detail

class ByteBuffer;
class ByteBufferView;

template<typename T>
using is_buffer = detail::is_buffer_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_buffer_view = detail::is_buffer_view_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_buffer_or_view = std::disjunction<is_buffer<T>, is_buffer_view<T>>;

template<typename T>
constexpr auto is_buffer_v = is_buffer<T>::value;

template<typename T>
constexpr auto is_buffer_view_v = is_buffer_view<T>::value;

template<typename T>
constexpr auto is_buffer_or_view_v = is_buffer_or_view<T>::value;

template<typename T>
using buffer_element = detail::buffer_element_impl<std::remove_cvref_t<T>>;

template<typename T>
using buffer_element_t = typename buffer_element<T>::type;

template<typename T>
class SparseBuffer;

template<typename T>
class BufferView;

// check if this data type is legitimate
template<typename T>
constexpr bool is_valid_buffer_element_v =
    std::is_same_v<T, std::remove_cvref_t<T>> &&
    std::is_trivially_copyable_v<T> &&
    std::is_trivially_destructible_v<T>;

/**
 * @brief Linear memory range on the device.
 *
 * Buffer<T> is a one-dimensional array of elements of type T located on the device.
 * It is strongly typed and ensures alignment based on the element type. Buffers
 * are the primary way to store and transfer structured data between host and device.
 * 
 * @tparam T Element type. Must be trivially copyable and destructible.
 *             Supported types include scalars (int, float), vectors (float2, float4),
 *             matrices, and user-defined structs reflected with LUISA_STRUCT.
 * 
 * Logic: Buffer is a Resource that manages device-side memory. It provides
 * methods to create command units (e.g., copy_to, copy_from) and views for 
 * sub-range access.
 *
 * @note Buffers follow RAII semantics. When a Buffer goes out of scope, the
 *       device memory is automatically scheduled for deallocation.
 * 
 * @see BufferView for non-owning sub-range views
 * 
 * Example:
 * @code
 * // Create a buffer of 1000 float4 elements
 * Buffer<float4> buffer = device.create_buffer<float4>(1000);
 * 
 * // Upload data from host
 * std::vector<float4> host_data(1000);
 * stream << buffer.copy_from(host_data.data()) << synchronize();
 * 
 * // Use in kernel
 * Kernel1D process = [&](BufferFloat4 buf) noexcept {
 *     Var idx = dispatch_id().x;
 *     Float4 value = buf.read(idx);
 *     buf.write(idx, value * 2.0f);
 * };
 * stream << device.compile(process)(buffer).dispatch(1000);
 * @endcode
 */
template<typename T>
class Buffer final : public Resource {

    static_assert(is_valid_buffer_element_v<T>);

private:
    size_t _size{};
    size_t _element_stride{};

private:
    friend class Device;
    friend class ResourceGenerator;
    friend class DxCudaInterop;
    friend class VkCudaInterop;
    friend class PinnedMemoryExt;
    Buffer(DeviceInterface *device, const BufferCreationInfo &info) noexcept
        : Resource{device, Tag::BUFFER, info},
          _size{info.total_size_bytes / info.element_stride},
          _element_stride{info.element_stride} {}
    Buffer(DeviceInterface *device, size_t size) noexcept
        : Buffer{device, [&] {
                     if (size == 0) [[unlikely]] {
                         detail::error_buffer_size_is_zero();
                     }
                     return device->create_buffer(Type::of<T>(), size, nullptr);
                 }()} {}

public:
    Buffer() noexcept = default;

    /**
     * @brief Destructor. Automatically destroys the device-side buffer.
     */
    ~Buffer() noexcept override {
        if (*this) { device()->destroy_buffer(handle()); }
    }

    Buffer(Buffer &&) noexcept = default;
    Buffer(Buffer const &) noexcept = delete;
    Buffer &operator=(Buffer &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Buffer &operator=(Buffer const &) noexcept = delete;
    using Resource::operator bool;
    using Resource::release;

    /**
     * @brief Get the number of elements in the buffer.
     * @return The element count.
     * @note This is the element count, not the byte size. Use size_bytes() for bytes.
     */
    [[nodiscard]] auto size() const noexcept {
        _check_is_valid();
        return _size;
    }

    /**
     * @brief Get the stride (byte size) of a single element.
     * @return Element size in bytes.
     * @note This is typically sizeof(T) but may be larger due to alignment requirements.
     */
    [[nodiscard]] constexpr auto stride() const noexcept {
        _check_is_valid();
        return _element_stride;
    }

    /**
     * @brief Get the total buffer size in bytes.
     * @return Total size = size() * stride().
     */
    [[nodiscard]] auto size_bytes() const noexcept {
        _check_is_valid();
        return _size * _element_stride;
    }

    /**
     * @brief Create a view of the entire buffer.
     * @return A BufferView object covering all elements.
     * 
     * Example:
     * @code
     * auto view = buffer.view();
     * stream << kernel(view).dispatch(view.size());
     * @endcode
     */
    [[nodiscard]] auto view() const noexcept {
        _check_is_valid();
        return BufferView<T>{this->native_handle(), this->handle(), _element_stride, 0u, _size, _size};
    }

    /**
     * @brief Create a view of a sub-range of the buffer.
     * @param offset Start index (in elements) of the sub-view.
     * @param count Number of elements in the sub-view.
     * @return A sub-range BufferView.
     * 
     * Example:
     * @code
     * // Process only the middle 1000 elements
     * auto subview = buffer.view(500, 1000);
     * stream << kernel(subview).dispatch(subview.size());
     * @endcode
     */
    [[nodiscard]] auto view(size_t offset, size_t count) const noexcept {
        return view().subview(offset, count);
    }

#ifndef LUISA_ENABLE_SAFE_MODE
    /**
     * @brief Create a command to download buffer data to the host.
     * @param data Host-side destination pointer.
     * @return A BufferDownloadCommand.
     */
    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return this->view().copy_to(data);
    }

    /**
     * @brief Create a command to copy data between buffers.
     */
    [[nodiscard]] auto copy_to(BufferView<T> dst) const noexcept {
        return this->view().copy_to(dst);
    }

    /**
     * @brief Create a command to upload data from host to buffer.
     * @param data Host-side source pointer.
     */
    [[nodiscard]] auto copy_from(const void *data) const noexcept {
        return this->view().copy_from(data);
    }
#endif

    /**
     * @brief DSL access to the buffer.
     * 
     * Logic: This allows using `buffer->read(index)` and `buffer->write(index, value)` 
     * within DSL kernels. It returns a proxy that records AST nodes.
     */
    [[nodiscard]] auto operator->() const noexcept {
        _check_is_valid();
        return reinterpret_cast<const detail::BufferExprProxy<Buffer<T>> *>(this);
    }
};

/**
 * @brief Non-owning reference to a Buffer.
 *
 * BufferView allows passing sub-ranges of buffers to shaders and commands
 * without transferring ownership. It is a lightweight handle that can be
 * created from a Buffer or another BufferView.
 * 
 * BufferViews are useful for:
 * - Processing sub-ranges of large buffers
 * - Passing buffer references to kernels without copying
 * - Reinterpreting buffer data as different types
 * 
 * @tparam T Element type of the buffer view.
 * 
 * Example:
 * @code
 * Buffer<float> large = device.create_buffer<float>(10000);
 * 
 * // Create a view of elements 1000-1999
 * auto view = large.view(1000, 1000);
 * 
 * // Use the view in a kernel
 * stream << kernel(view).dispatch(view.size());
 * @endcode
 * 
 * @see Buffer
 */
template<typename T>
class BufferView {
    friend class lc::validation::Stream;
    static_assert(is_valid_buffer_element_v<T>);

private:
    void *_native_handle;
    uint64_t _handle;
    size_t _offset_bytes;
    size_t _element_stride;
    size_t _size;
    size_t _total_size;

private:
    friend class Buffer<T>;
    friend class SparseBuffer<T>;

    template<typename U>
    friend class BufferView;

public:
    BufferView(void *native_handle, uint64_t handle,
               size_t element_stride, size_t offset_bytes,
               size_t size, size_t total_size) noexcept
        : _native_handle{native_handle}, _handle{handle}, _offset_bytes{offset_bytes},
          _element_stride{element_stride}, _size{size}, _total_size{total_size} {
        if (_offset_bytes % alignof(T) != 0u) [[unlikely]] {
            detail::error_buffer_invalid_alignment(_offset_bytes, alignof(T));
        }
    }

    template<template<typename> typename B>
        requires(is_buffer_v<B<T>>)
    BufferView(const B<T> &buffer) noexcept : BufferView{buffer.view()} {}

    BufferView() noexcept : BufferView{nullptr, invalid_resource_handle, 0, 0, 0, 0} {}
    [[nodiscard]] explicit operator bool() const noexcept { return _handle != invalid_resource_handle; }

    // properties
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto native_handle() const noexcept { return _native_handle; }
    [[nodiscard]] constexpr auto stride() const noexcept { return _element_stride; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto offset() const noexcept { return _offset_bytes / _element_stride; }
    [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * _element_stride; }
    [[nodiscard]] auto total_size() const noexcept { return _total_size; }
    [[nodiscard]] auto total_size_bytes() const noexcept { return _total_size * _element_stride; }

    [[nodiscard]] auto original() const noexcept {
        return BufferView{_native_handle, _handle,
                          _element_stride, 0u,
                          _total_size, _total_size};
    }
    [[nodiscard]] auto subview(size_t offset_elements, size_t size_elements) const noexcept {
        if (size_elements + offset_elements > _size) [[unlikely]] {
            detail::error_buffer_subview_overflow(offset_elements, size_elements, _size);
        }
        return BufferView{_native_handle, _handle, _element_stride,
                          _offset_bytes + offset_elements * _element_stride,
                          size_elements, _total_size};
    }
    /**
     * @brief Reinterpret the buffer view as a different element type.
     * @tparam U Target element type.
     * @return A BufferView<U> with the same underlying memory.
     * 
     * This allows viewing the same memory with different types, similar to
     * reinterpret_cast in C++. The total byte size must be compatible.
     * 
     * Example:
     * @code
     * Buffer<float> float_buf = device.create_buffer<float>(1000);
     * // View as bytes
     * auto byte_view = float_buf.view().as<std::byte>();
     * // View as float4 (250 elements)
     * auto vec4_view = float_buf.view().as<float4>();
     * @endcode
     */
    template<typename U>
        requires(!is_custom_struct_v<U>)
    [[nodiscard]] auto as() const noexcept {
        if (this->size_bytes() < sizeof(U)) [[unlikely]] {
            detail::error_buffer_reinterpret_size_too_small(sizeof(U), this->size_bytes());
        }
        auto total_size_bytes = _total_size * _element_stride;
        return BufferView<U>{_native_handle, _handle, sizeof(U), _offset_bytes,
                             this->size_bytes() / sizeof(U), total_size_bytes / sizeof(U)};
    }
    // commands
    // copy buffer's data to pointer
    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return luisa::make_unique<BufferDownloadCommand>(_handle, offset_bytes(), size_bytes(), data);
    }
    // copy buffer's data to another buffer
    [[nodiscard]] auto copy_to(BufferView<T> dst) const noexcept {
        return dst.copy_from(*this);
    }
    // copy buffer's data to a byte buffer
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_to(const ByteBufferView &dst) const noexcept;
    // copy pointer's data to buffer
    [[nodiscard]] auto copy_from(const void *data) const noexcept {
        return luisa::make_unique<BufferUploadCommand>(this->handle(), this->offset_bytes(), this->size_bytes(), data);
    }
    // copy source buffer's data to buffer
    [[nodiscard]] auto copy_from(BufferView<T> source) const noexcept {
        if (source.size() != this->size()) [[unlikely]] {
            detail::error_buffer_copy_sizes_mismatch(source.size(), this->size());
        }
        return luisa::make_unique<BufferCopyCommand>(
            source.handle(), this->handle(),
            source.offset_bytes(), this->offset_bytes(),
            this->size_bytes());
    }
    // copy source byte buffer's data to buffer
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_from(const ByteBufferView &source) const noexcept;
    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::BufferExprProxy<BufferView<T>> *>(this);
    }
};

template<typename T>
BufferView(const Buffer<T> &) -> BufferView<T>;

template<typename T>
BufferView(BufferView<T>) -> BufferView<T>;

namespace detail {

template<typename T>
struct is_buffer_impl<Buffer<T>> : std::true_type {};

template<typename T>
struct is_buffer_view_impl<BufferView<T>> : std::true_type {};

template<>
struct is_buffer_impl<ByteBuffer> : std::true_type {};

template<>
struct is_buffer_view_impl<ByteBufferView> : std::true_type {};

template<typename T>
struct buffer_element_impl<Buffer<T>> {
    using type = T;
};

template<typename T>
struct buffer_element_impl<BufferView<T>> {
    using type = T;
};

}// namespace detail

}// namespace luisa::compute
