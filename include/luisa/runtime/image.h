#pragma once

#include <luisa/core/mathematics.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/mipmap.h>
#include <luisa/runtime/rhi/sampler.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

namespace detail {

template<typename ImageOrView>
class ImageExprProxy;

LUISA_RUNTIME_API void error_image_invalid_mip_levels(size_t level, size_t mip) noexcept;
LUISA_RUNTIME_API void image_size_zero_error() noexcept;

}// namespace detail

template<typename T>
class ImageView;

template<typename T>
class SparseImage;

template<typename T>
class BufferView;

class BindlessArray;

/**
 * @brief 2D texture resource on the device.
 *
 * Image<T> represents a 2D grid of pixels stored on the device. 
 * Unlike Buffers, Images are optimized for 2D spatial locality and support 
 * hardware-accelerated sampling with automatic format conversion.
 * 
 * @tparam T Pixel channel type (float, int, or uint). This determines how
 *             pixels are read/written in DSL, not the internal storage format.
 * 
 * Logic: Image is a Resource mapped to a backend texture object. The internal
 * storage format (e.g., BYTE4, FLOAT4) is specified at creation time, and
 * automatic conversion happens when reading/writing pixel values.
 * 
 * @note Images follow RAII semantics. When an Image goes out of scope, the
 *       device memory is automatically scheduled for deallocation.
 * 
 * @see ImageView for accessing specific mipmap levels
 * @see Volume for 3D textures
 * 
 * Example:
 * @code
 * // Create a 1920x1080 RGBA image with 8-bit storage
 * Image<float> image = device.create_image<float>(
 *     PixelStorage::BYTE4, 1920, 1080);
 * 
 * // Upload pixel data from host
 * std::vector<std::byte> pixels(1920 * 1080 * 4);
 * stream << image.copy_from(pixels.data()) << synchronize();
 * 
 * // Use in kernel
 * Kernel2D process = [&](ImageFloat img) noexcept {
 *     Var coord = dispatch_id().xy();
 *     Float4 color = img.read(coord);
 *     img.write(coord, color * 0.5f);  // Darken
 * };
 * stream << device.compile(process)(image).dispatch(1920, 1080);
 * @endcode
 */
template<typename T>
class Image final : public Resource {
    static_assert(is_legal_image_element<T>);

private:
    uint2 _size{};
    uint32_t _mip_levels{};
    PixelStorage _storage{};

private:
    friend class Device;
    friend class ResourceGenerator;
    friend class DxCudaInterop;
    friend class VkCudaInterop;
    Image(DeviceInterface *device,
          const ResourceCreationInfo &create_info,
          PixelStorage storage,
          uint2 size, uint mip_levels) noexcept
        : Resource{device, Tag::TEXTURE, create_info},
          _size{size},
          _mip_levels{detail::max_mip_levels(make_uint3(size, 1u), mip_levels)},
          _storage{storage} {
        if (size.x == 0 || size.y == 0) [[unlikely]] {
            detail::image_size_zero_error();
        }
    }

public:
    Image() noexcept = default;

    /**
     * @brief Destructor. Automatically destroys the device-side texture.
     */
    ~Image() noexcept override {
        if (*this) { device()->destroy_texture(handle()); }
    }

    using Resource::operator bool;
    using Resource::release;
    Image(Image &&) noexcept = default;
    Image(Image const &) noexcept = delete;
    Image &operator=(Image &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Image &operator=(Image const &) noexcept = delete;

    /**
     * @brief Get the image dimensions.
     * @return uint2 containing (width, height) in pixels.
     */
    [[nodiscard]] auto size() const noexcept {
        _check_is_valid();
        return _size;
    }

    /**
     * @brief Get the number of mipmap levels.
     * @return Mipmap count (1 for no mipmaps, >1 for mipmapped images).
     */
    [[nodiscard]] auto mip_levels() const noexcept {
        _check_is_valid();
        return _mip_levels;
    }

    /**
     * @brief Get the internal storage format.
     * @return PixelStorage enum value (e.g., BYTE4, FLOAT4).
     * @see PixelStorage
     */
    [[nodiscard]] auto storage() const noexcept {
        _check_is_valid();
        return _storage;
    }

    /**
     * @brief Get the backend-specific pixel format.
     * @return PixelFormat enum value.
     * @note This is the actual format used by the GPU driver.
     */
    [[nodiscard]] auto format() const noexcept {
        _check_is_valid();
        return pixel_storage_to_format<T>(_storage);
    }

    /**
     * @brief Create a view of a specific mipmap level.
     * @param level Mipmap level index (0 = full resolution).
     * @return An ImageView object for that mipmap level.
     * @note Level 0 is the base level (full resolution).
     *       Each subsequent level is half the size of the previous.
     */
    [[nodiscard]] auto view(uint32_t level) const noexcept {
        _check_is_valid();
        if (level >= _mip_levels) [[unlikely]] {
            detail::error_image_invalid_mip_levels(level, _mip_levels);
        }
        auto mip_size = luisa::max(_size >> level, 1u);
        return ImageView<T>{native_handle(), handle(), _storage, level, mip_size};
    }

    /**
     * @brief Create a view of the base mipmap level (level 0).
     * @return An ImageView for the full-resolution image.
     * @note This is equivalent to view(0).
     */
    [[nodiscard]] auto view() const noexcept { return view(0u); }

    /**
     * @brief Create a command to download image data to the host.
     * @tparam U Destination type (pointer or another Image/Buffer).
     * @return A command object for use with operator<< on a Stream.
     * 
     * Example:
     * @code
     * std::vector<std::byte> pixels(image.size().x * image.size().y * 4);
     * stream << image.copy_to(pixels.data()) << synchronize();
     * @endcode
     */
    template<typename U>
    [[nodiscard]] auto copy_to(U &&dst) const noexcept {
        _check_is_valid();
        return this->view(0).copy_to(std::forward<U>(dst));
    }

    /**
     * @brief Create a command to upload image data from the host.
     * @tparam U Source type (pointer or another Image/Buffer).
     * @return A command object for use with operator<< on a Stream.
     */
    template<typename U>
    [[nodiscard]] auto copy_from(U &&dst) const noexcept {
        _check_is_valid();
        return this->view(0).copy_from(std::forward<U>(dst));
    }

    /**
     * @brief DSL access to the image.
     * 
     * Logic: This allows using `image->read(coord)` and `image->write(coord, value)`
     * within DSL kernels.
     */
    [[nodiscard]] auto operator->() const noexcept {
        _check_is_valid();
        return reinterpret_cast<const detail::ImageExprProxy<Image<T>> *>(this);
    }
};

/**
 * @brief Non-owning reference to an Image mipmap level.
 * 
 * ImageView provides access to a specific mipmap level of an Image.
 * It is a lightweight handle that can be passed to kernels and commands
 * without transferring ownership.
 * 
 * ImageViews are useful for:
 * - Processing specific mipmap levels
 * - Passing image references to kernels
 * - Creating sub-views (for future extensions)
 * 
 * @tparam T Pixel channel type (float, int, or uint).
 * 
 * @see Image
 * @see VolumeView for 3D texture views
 * 
 * Example:
 * @code
 * Image<float> mip_image = device.create_image<float>(
 *     PixelStorage::BYTE4, 1024, 1024, 10);  // With mipmaps
 * 
 * // Access level 2 (256x256)
 * auto view = mip_image.view(2);
 * stream << kernel(view).dispatch(view.size().x, view.size().y);
 * @endcode
 */
template<typename T>
class ImageView {

private:
    void *_native_handle;
    uint64_t _handle;
    uint2 _size;
    uint _level;
    PixelStorage _storage;

private:
    friend class Image<T>;
    friend class SparseImage<T>;
    friend class detail::MipmapView;
    friend class DepthBuffer;

    [[nodiscard]] auto _as_mipmap() const noexcept {
        return detail::MipmapView{_handle, make_uint3(_size, 1u), _level, _storage};
    }

public:
    ImageView(void *native_handle, uint64_t handle, PixelStorage storage,
              uint level, uint2 size) noexcept
        : _native_handle{native_handle},
          _handle{handle}, _size{size},
          _level{level}, _storage{storage} {}

    ImageView(const Image<T> &image) noexcept : ImageView{image.view(0u)} {}
    // properties
    [[nodiscard]] auto native_handle() const noexcept { return _native_handle; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto size_bytes() const noexcept {
        return pixel_storage_size(_storage, make_uint3(_size, 1u));
    }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto format() const noexcept { return pixel_storage_to_format<T>(_storage); }
    [[nodiscard]] auto level() const noexcept { return _level; }
    // commands
    // copy image's data to pointer or another image
    template<typename U>
    [[nodiscard]] auto copy_to(U &&dst) const noexcept { return _as_mipmap().copy_to(std::forward<U>(dst)); }
    // copy pointer or another image's data to image
    template<typename U>
    [[nodiscard]] auto copy_from(U &&src) const noexcept { return _as_mipmap().copy_from(std::forward<U>(src)); }
    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::ImageExprProxy<ImageView<T>> *>(this);
    }
};

template<typename T>
ImageView(const Image<T> &) -> ImageView<T>;

template<typename T>
ImageView(ImageView<T>) -> ImageView<T>;

namespace detail {

template<typename T>
struct is_image_impl : std::false_type {};

template<typename T>
struct is_image_impl<Image<T>> : std::true_type {};

template<typename T>
struct is_image_view_impl : std::false_type {};

template<typename T>
struct is_image_view_impl<ImageView<T>> : std::true_type {};

}// namespace detail

template<typename T>
using is_image = detail::is_image_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_image_view = detail::is_image_view_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_image_or_view = std::disjunction<is_image<T>, is_image_view<T>>;

template<typename T>
constexpr auto is_image_v = is_image<T>::value;

template<typename T>
constexpr auto is_image_view_v = is_image_view<T>::value;

template<typename T>
constexpr auto is_image_or_view_v = is_image_or_view<T>::value;

namespace detail {

template<typename ImageOrView>
struct image_element_impl {
    static_assert(always_false_v<ImageOrView>);
};

template<typename T>
struct image_element_impl<Image<T>> {
    using type = T;
};

template<typename T>
struct image_element_impl<ImageView<T>> {
    using type = T;
};

}// namespace detail

template<typename ImageOrView>
using image_element = detail::image_element_impl<std::remove_cvref_t<ImageOrView>>;

template<typename ImageOrView>
using image_element_t = typename image_element<ImageOrView>::type;

}// namespace luisa::compute
