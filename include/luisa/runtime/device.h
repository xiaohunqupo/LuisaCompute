#pragma once

#ifdef LUISA_ENABLE_IR
#include <luisa/ir/ir2ast.h>
#endif
#include <luisa/ast/type_registry.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa {
class BinaryIO;
}// namespace luisa

namespace luisa::compute {

class Context;
class Event;
class TimelineEvent;
class Stream;
class Mesh;
class Curve;
class MotionInstance;
class MeshFormat;
class ProceduralPrimitive;
class Accel;
class Swapchain;
class BindlessArray;
class IndirectDispatchBuffer;
class SparseBufferHeap;
class SparseTextureHeap;
class ByteBuffer;

template<typename T>
class SOA;

template<typename T>
class Buffer;

template<typename T>
class SparseBuffer;

template<typename T>
class Image;

template<typename T>
class Volume;

template<size_t dimension, concepts::non_cvref... Args>
class Shader;

template<size_t dim, typename... Args>
class AOTShader;

template<size_t N, typename... Args>
class Kernel;

template<typename... Args>
class RasterShader;

template<typename VertCallable, typename PixelCallable>
class RasterKernel;

template<typename... Args>
struct Kernel1D;

template<typename... Args>
struct Kernel2D;

template<typename... Args>
struct Kernel3D;

template<typename T>
class SparseImage;

template<typename T>
class SparseVolume;

class DepthBuffer;

namespace detail {

template<typename T>
struct is_dsl_kernel : std::false_type {};

template<size_t N, typename... Args>
struct is_dsl_kernel<Kernel<N, Args...>> : std::true_type {};

template<typename... Args>
struct is_dsl_kernel<Kernel1D<Args...>> : std::true_type {};

template<typename... Args>
struct is_dsl_kernel<Kernel2D<Args...>> : std::true_type {};

template<typename... Args>
struct is_dsl_kernel<Kernel3D<Args...>> : std::true_type {};

}// namespace detail

template<typename T>
concept device_extension = std::is_base_of_v<DeviceExtension, T> &&
                           std::is_same_v<const luisa::string_view, decltype(T::name)>;

/**
 * @brief Device abstraction for high-performance computation.
 *
 * The Device class represents a logical computation unit (e.g., a GPU).
 * It is responsible for:
 * 1. Creating and managing device resources (Buffers, Images, etc.).
 * 2. Compiling Kernels into executable Shaders.
 * 3. Creating Streams for asynchronous command execution.
 * 
 * Logic: Device is a wrapper around a backend-specific DeviceInterface.
 * Most of its methods delegate calls to the underlying implementation.
 */
class LUISA_RUNTIME_API Device {

public:
    using Deleter = void(DeviceInterface *);
    using Creator = DeviceInterface *(Context && /* context */, const DeviceConfig * /* properties */);
    using Handle = luisa::shared_ptr<DeviceInterface>;

private:
    Handle _impl;

    template<typename T, typename... Args>
    [[nodiscard]] auto _create(Args &&...args) noexcept {
        return T{this->_impl.get(), std::forward<Args>(args)...};
    }

    static void _check_no_implicit_binding(Function func, luisa::string_view shader_path) noexcept;

public:
    Device() noexcept = default;

    /**
     * @brief Construct Device from a backend handle.
     * @param handle Shared pointer to the device interface.
     */
    explicit Device(Handle handle) noexcept : _impl{std::move(handle)} {}

    /// @return Name of the backend (e.g., "cuda", "dx").
    [[nodiscard]] auto backend_name() const noexcept { return _impl->backend_name(); }

    /// @return Native driver handle (e.g., CUcontext or ID3D12Device*).
    [[nodiscard]] auto native_handle() const noexcept { return _impl->native_handle(); }

    /// @return Pointer to the underlying device interface.
    [[nodiscard]] auto impl() const noexcept { return _impl.get(); }
    [[nodiscard]] auto const &impl_shared() const & noexcept { return _impl; }
    [[nodiscard]] auto &&impl_shared() && noexcept { return std::move(_impl); }

    /// @return Warp size (simd width) of the compute units.
    [[nodiscard]] auto compute_warp_size() const noexcept { return _impl->compute_warp_size(); }

    /// @return Alignment requirement for memory allocations.
    [[nodiscard]] auto memory_granularity() const noexcept { return _impl->memory_granularity(); }

    /// @return True if the device is properly initialized.
    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(_impl); }

    /**
     * @brief Access backend-specific extensions.
     * @tparam Ext Extension type.
     * @return Pointer to the requested extension, or nullptr if not supported.
     */
    template<device_extension Ext>
    [[nodiscard]] auto extension() const noexcept {
        return static_cast<Ext *>(_impl->extension(Ext::name));
    }

    /**
     * @brief Create a command stream.
     * @param stream_tag Type of the stream (Compute, Graphics, Transfer).
     * @return A Stream object for submitting commands.
     */
    [[nodiscard]] Stream create_stream(StreamTag stream_tag = StreamTag::COMPUTE) noexcept;

    /**
     * @brief Create an event for synchronization.
     * @return An Event object.
     */
    [[nodiscard]] Event create_event() noexcept;

    /**
     * @brief Create a timeline event for fine-grained synchronization.
     * @return A TimelineEvent object.
     */
    [[nodiscard]] TimelineEvent create_timeline_event() noexcept;

    /**
     * @brief Create a swapchain for window presentation.
     * @param stream The stream used for presentation.
     * @param option Swapchain configuration options.
     * @return A Swapchain object.
     */
    [[nodiscard]] Swapchain create_swapchain(const Stream &stream, const SwapchainOption &option) noexcept;

    /**
     * @brief Create a buffer for indirect dispatch.
     * @param capacity Maximum number of dispatches the buffer can hold.
     * @return An IndirectDispatchBuffer object.
     */
    [[nodiscard]] IndirectDispatchBuffer create_indirect_dispatch_buffer(size_t capacity) noexcept;

    /**
     * @brief Create a ray-tracing mesh.
     * @param vertices Buffer containing vertex data.
     * @param triangles Buffer containing triangle indices.
     * @param option Mesh acceleration structure options.
     * @return A Mesh object.
     */
    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] Mesh create_mesh(VBuffer &&vertices,
                                   TBuffer &&triangles,
                                   const AccelOption &option = {}) noexcept;

    /**
     * @brief Create a ray-tracing mesh with custom vertex stride.
     */
    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] Mesh create_mesh(VBuffer &&vertices,
                                   size_t vertex_stride,
                                   TBuffer &&triangles,
                                   const AccelOption &option = {}) noexcept;

    /**
     * @brief Create a ray-tracing curve.
     */
    template<typename CPBuffer, typename SegmentBuffer>
    [[nodiscard]] Curve create_curve(CurveBasis basis,
                                     CPBuffer &&control_points,
                                     SegmentBuffer &&segments,
                                     const AccelOption &option = {}) noexcept;

    /**
     * @brief Create procedural primitives for ray tracing.
     */
    template<typename AABBBuffer>
    [[nodiscard]] ProceduralPrimitive create_procedural_primitive(AABBBuffer &&aabb_buffer,
                                                                  const AccelOption &option = {}) noexcept;

    /**
     * @brief Create a motion instance for temporal ray tracing.
     */
    [[nodiscard]] MotionInstance create_motion_instance(const Mesh &mesh, const AccelMotionOption &option) noexcept;
    [[nodiscard]] MotionInstance create_motion_instance(const Curve &curve, const AccelMotionOption &option) noexcept;
    [[nodiscard]] MotionInstance create_motion_instance(const ProceduralPrimitive &primitive, const AccelMotionOption &option) noexcept;

    /**
     * @brief Create a top-level acceleration structure (Accel).
     * @param option Accel configuration options.
     * @return An Accel object.
     */
    [[nodiscard]] Accel create_accel(const AccelOption &option = {}) noexcept;

    /**
     * @brief Create a bindless array for resource indexing in shaders.
     * @param slot_count Number of slots in the array.
     * @param type Resource types allowed in the slots.
     * @return A BindlessArray object.
     */
    [[nodiscard]] BindlessArray create_bindless_array(size_t slot_count = 65536u, BindlessSlotType type = BindlessSlotType::MULTIPLE) noexcept;

    /**
     * @brief Create a 2D image.
     * @tparam T Pixel channel type (e.g., float, int).
     * @param pixel Internal storage format.
     * @param width Width in pixels.
     * @param height Height in pixels.
     * @param mip_levels Number of mipmap levels.
     * @return An Image object.
     */
    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint width, uint height, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return _create<Image<T>>(pixel, make_uint2(width, height), mip_levels, simultaneous_access, allow_raster_target);
    }

    /**
     * @brief Create a 3D volume.
     */
    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint width, uint height, uint depth, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return _create<Volume<T>>(pixel, make_uint3(width, height, depth), mip_levels, simultaneous_access, allow_raster_target);
    }

    /**
     * @brief Create a linear buffer.
     * @tparam T Element type.
     * @param size Number of elements.
     * @return A Buffer object.
     */
    template<typename T>
        requires(!is_custom_struct_v<T>)
    [[nodiscard]] auto create_buffer(size_t size) noexcept {
        return _create<Buffer<T>>(size);
    }

    /**
     * @brief Compile a kernel into a shader.
     * @param kernel The DSL kernel definition.
     * @param option Compilation options (FastMath, Cache, etc.).
     * @return A runnable Shader object.
     * 
     * Logic: This triggers the backend compilation pipeline:
     * 1. Trace kernel to build AST.
     * 2. Backend translates AST to platform-dependent source.
     * 3. Backend calls driver compiler (NVCC, DXC, etc.) to generate binary.
     */
    template<size_t N, typename... Args>
    [[nodiscard]] auto compile(const Kernel<N, Args...> &kernel,
                               const ShaderOption &option = {}) noexcept {
        return _create<Shader<N, Args...>>(kernel.function()->function(), option);
    }

    template<typename Kernel>
    void compile_to(Kernel &&kernel,
                    luisa::string_view name,
                    bool enable_fast_math = true,
                    bool enable_debug_info = false) noexcept {
        ShaderOption option{
            .enable_cache = false,
            .enable_fast_math = enable_fast_math,
            .enable_debug_info = enable_debug_info,
            .compile_only = true,
            .name = luisa::string{name}};
        static_cast<void>(this->compile(std::forward<Kernel>(kernel), option));
    }

    template<size_t N, typename Func>
        requires(std::negation_v<detail::is_dsl_kernel<std::remove_cvref_t<Func>>> && N >= 1 && N <= 3)
    [[nodiscard]] auto compile(Func &&f, const ShaderOption &option = {}) noexcept {
        if constexpr (N == 1u) {
            return compile(Kernel1D{std::forward<Func>(f)}, option);
        } else if constexpr (N == 2u) {
            return compile(Kernel2D{std::forward<Func>(f)}, option);
        } else {
            return compile(Kernel3D{std::forward<Func>(f)}, option);
        }
    }

    template<size_t N, typename Kernel>
    void compile_to(Kernel &&kernel,
                    luisa::string_view name,
                    bool enable_fast_math = true,
                    bool enable_debug_info = false) noexcept {
        ShaderOption option{
            .enable_cache = false,
            .enable_fast_math = enable_fast_math,
            .enable_debug_info = enable_debug_info,
            .compile_only = true,
            .name = luisa::string{name}};
        static_cast<void>(this->compile<N>(std::forward<Kernel>(kernel), option));
    }

#ifdef LUISA_ENABLE_IR
    template<size_t N, typename... Args>
    [[nodiscard]] auto compile(const ir::KernelModule *const module,
                               const ShaderOption &option = {}) noexcept {
        return _create<Shader<N, Args...>>(module, option);
    }
#endif

    template<typename V, typename P>
    [[nodiscard]] typename RasterKernel<V, P>::RasterShaderType compile(
        const RasterKernel<V, P> &kernel,
        const MeshFormat &mesh_format,
        const ShaderOption &option = {}) noexcept;

    template<typename V, typename P>
    void compile_to(
        const RasterKernel<V, P> &kernel,
        const MeshFormat &mesh_format,
        luisa::string_view serialization_path,
        const ShaderOption &option = {}) noexcept;

    template<typename... Args>
    RasterShader<Args...> load_raster_shader(
        luisa::string_view shader_name) noexcept;

    template<size_t N, typename... Args>
    [[nodiscard]] auto load_shader(luisa::string_view shader_name) noexcept {
        return _create<Shader<N, Args...>>(shader_name);
    }

    [[nodiscard]] auto query(std::string_view meta_expr) const noexcept {
        return _impl->query(meta_expr);
    }

    template<typename T, typename... Args>
    [[nodiscard]] auto create(Args &&...args) noexcept {
        return _create<T>(std::forward<Args>(args)...);
    }
};

}// namespace luisa::compute
