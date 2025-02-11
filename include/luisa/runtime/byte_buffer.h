#pragma once

#include <luisa/runtime/buffer.h>

namespace luisa::compute {

namespace detail {
class ByteBufferExprProxy;
}// namespace detail
class ByteBufferView;

class LC_RUNTIME_API ByteBuffer final : public Resource {

private:
    size_t _size_bytes{};

private:
    friend class Device;
    friend class ResourceGenerator;
    ByteBuffer(DeviceInterface *device, const BufferCreationInfo &info) noexcept;
    ByteBuffer(DeviceInterface *device, size_t size_bytes) noexcept;

public:
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }
    ByteBuffer() noexcept = default;
    ~ByteBuffer() noexcept override;
    ByteBuffer(ByteBuffer &&) noexcept = default;
    ByteBuffer(ByteBuffer const &) noexcept = delete;
    ByteBuffer &operator=(ByteBuffer &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    ByteBuffer &operator=(ByteBuffer const &) noexcept = delete;
    using Resource::operator bool;
    [[nodiscard]] auto copy_to(void *data) const noexcept {
        _check_is_valid();
        return luisa::make_unique<BufferDownloadCommand>(handle(), 0u, _size_bytes, data);
    }
    [[nodiscard]] auto copy_from(const void *data) noexcept {
        _check_is_valid();
        return luisa::make_unique<BufferUploadCommand>(handle(), 0u, _size_bytes, data);
    }
    [[nodiscard]] auto copy_from(const void *data, size_t buffer_offset, size_t size_bytes) noexcept {
        _check_is_valid();
        if (size_bytes > _size_bytes) [[unlikely]] {
            detail::error_buffer_copy_sizes_mismatch(size_bytes, _size_bytes);
        }
        return luisa::make_unique<BufferUploadCommand>(handle(), buffer_offset, size_bytes, data);
    }
    template<typename T>
    [[nodiscard]] auto copy_from(BufferView<T> source) noexcept {
        _check_is_valid();
        if (source.size_bytes() != _size_bytes) [[unlikely]] {
            detail::error_buffer_copy_sizes_mismatch(source.size_bytes(), _size_bytes);
        }
        return luisa::make_unique<BufferCopyCommand>(
            source.handle(), this->handle(),
            source.offset_bytes(), 0u,
            this->size_bytes());
    }
    [[nodiscard]] auto copy_from(const ByteBuffer &source, size_t offset, size_t size_bytes) noexcept {
        _check_is_valid();
        if (size_bytes > _size_bytes) [[unlikely]] {
            detail::error_buffer_copy_sizes_mismatch(size_bytes, _size_bytes);
        }
        return luisa::make_unique<BufferCopyCommand>(
            source.handle(), this->handle(),
            offset, 0u,
            size_bytes);
    }
    [[nodiscard]] ByteBufferView view() const noexcept;
    template<typename T>
    [[nodiscard]] auto copy_from(ByteBufferView source) const noexcept;
    [[nodiscard]] auto copy_from(ByteBufferView source, size_t offset, size_t size_bytes) const noexcept;
    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        _check_is_valid();
        return reinterpret_cast<const detail::ByteBufferExprProxy *>(this);
    }
};

class ByteBufferView {
    friend class lc::validation::Stream;

private:
    void *_native_handle;
    uint64_t _handle;
    size_t _offset_bytes;
    size_t _size;
    size_t _total_size;

public:
    ByteBufferView(void *native_handle, uint64_t handle,
                   size_t offset_bytes,
                   size_t size, size_t total_size) noexcept
        : _native_handle{native_handle}, _handle{handle}, _offset_bytes{offset_bytes},
          _size{size}, _total_size{total_size} {
    }

    ByteBufferView(const ByteBuffer &buffer) noexcept : ByteBufferView{buffer.view()} {}

    template<typename T>
        requires(is_buffer_v<T> && !std::is_same_v<ByteBuffer, std::remove_cvref_t<T>>)
    ByteBufferView(T const &buffer) : ByteBufferView{
                                          buffer.native_handle(), buffer.handle(),
                                          0, buffer.size_bytes(), buffer.size_bytes()} {}
    template<typename T>
        requires(is_buffer_view_v<T> && !std::is_same_v<ByteBufferView, std::remove_cvref_t<T>>)
    ByteBufferView(T buffer_view) : ByteBufferView{
                                          buffer_view.native_handle(), buffer_view.handle(),
                                          buffer_view.offset_bytes(), buffer_view.size_bytes(), buffer_view.total_size_bytes()} {}

    ByteBufferView() noexcept : ByteBufferView{nullptr, invalid_resource_handle, 0, 0, 0} {}
    [[nodiscard]] explicit operator bool() const noexcept { return _handle != invalid_resource_handle; }

    // properties
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto native_handle() const noexcept { return _native_handle; }
    [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size; }
    [[nodiscard]] auto total_size_bytes() const noexcept { return _total_size; }

    [[nodiscard]] auto original() const noexcept {
        return ByteBufferView{_native_handle, _handle,
                              0u, _total_size, _total_size};
    }
    [[nodiscard]] auto subview(size_t offset_bytes, size_t size_bytes) const noexcept {
        if (offset_bytes + size_bytes > _size) [[unlikely]] {
            detail::error_buffer_subview_overflow(offset_bytes, size_bytes, _size);
        }
        return ByteBufferView{_native_handle, _handle, _offset_bytes + offset_bytes,
                              size_bytes, _total_size};
    }

    template<typename U>
        requires(!is_custom_struct_v<U>)
    [[nodiscard]] auto as() const noexcept {
        if (this->size_bytes() < sizeof(U)) [[unlikely]] {
            detail::error_buffer_reinterpret_size_too_small(sizeof(U), this->size_bytes());
        }
        return BufferView<U>{_native_handle, _handle, sizeof(U), _offset_bytes,
                             this->size_bytes() / sizeof(U), _total_size / sizeof(U)};
    }
    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return luisa::make_unique<BufferDownloadCommand>(_handle, offset_bytes(), size_bytes(), data);
    }
    // copy pointer's data to buffer
    [[nodiscard]] auto copy_from(const void *data) const noexcept {
        return luisa::make_unique<BufferUploadCommand>(this->handle(), this->offset_bytes(), this->size_bytes(), data);
    }
    // copy source buffer's data to buffer
    [[nodiscard]] auto copy_from(ByteBufferView source) const noexcept {
        if (source.size_bytes() != this->size_bytes()) [[unlikely]] {
            detail::error_buffer_copy_sizes_mismatch(source.size_bytes(), this->size_bytes());
        }
        return luisa::make_unique<BufferCopyCommand>(
            source.handle(), this->handle(),
            source.offset_bytes(), this->offset_bytes(),
            this->size_bytes());
    }
};

template<typename T>
[[nodiscard]] inline auto ByteBuffer::copy_from(ByteBufferView source) const noexcept {
    _check_is_valid();
    if (source.size_bytes() != _size_bytes) [[unlikely]] {
        detail::error_buffer_copy_sizes_mismatch(source.size_bytes(), _size_bytes);
    }
    return luisa::make_unique<BufferCopyCommand>(
        source.handle(), this->handle(),
        source.offset_bytes(), 0u,
        this->size_bytes());
}
[[nodiscard]] inline auto ByteBuffer::copy_from(ByteBufferView source, size_t offset, size_t size_bytes) const noexcept {
    _check_is_valid();
    if (size_bytes > _size_bytes) [[unlikely]] {
        detail::error_buffer_copy_sizes_mismatch(size_bytes, _size_bytes);
    }
    return luisa::make_unique<BufferCopyCommand>(
        source.handle(), this->handle(),
        offset, 0u,
        size_bytes);
}

namespace detail {
LC_RUNTIME_API void error_buffer_size_not_aligned(size_t align) noexcept;
template<>
struct is_buffer_impl<ByteBuffer> : std::true_type {};
}// namespace detail

}// namespace luisa::compute
