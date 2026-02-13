#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/filesystem.h>

namespace luisa {
class DynamicModule;
class BinaryIO;
}// namespace luisa

namespace luisa::compute {

class Device;
struct DeviceConfig;

namespace detail {
class ContextImpl;
}// namespace detail

/**
 * @brief Context manages backend plug-ins and global runtime information.
 *
 * The Context class is the entry point of the LuisaCompute runtime. It is responsible
 * for locating and loading backend modules (like CUDA, DirectX, Metal), and
 * providing paths for data and runtime directories.
 */
class LUISA_RUNTIME_API Context {

private:
    luisa::shared_ptr<detail::ContextImpl> _impl;

public:
    /**
     * @brief Construct a Context with a shared implementation.
     * @param impl The implementation pointer.
     */
    explicit Context(luisa::shared_ptr<detail::ContextImpl> impl) noexcept;

    /**
     * @brief Construct a Context using the program path.
     * @param program_path The path to the current executable, used to locate backend plug-ins.
     * 
     * Logic: The context will look for backends in the same directory as the executable.
     */
    explicit Context(luisa::string_view program_path) noexcept;

    /**
     * @brief Construct a Context with program path and data directory.
     * @param program_path The path to the current executable.
     * @param data_dir The directory used for storing caches and internal data.
     */
    explicit Context(luisa::string_view program_path, luisa::string_view data_dir) noexcept;

    /**
     * @brief Construct a Context from a C-string program path.
     * @param program_path The path to the current executable.
     */
    explicit Context(const char *program_path) noexcept
        : Context{luisa::string_view{program_path}} {}

    ~Context() noexcept;
    Context(Context &&) noexcept = default;
    Context(const Context &) noexcept = default;
    Context &operator=(Context &&) noexcept = default;
    Context &operator=(const Context &) noexcept = default;

    /// @return The internal implementation pointer.
    [[nodiscard]] const auto &impl() const & noexcept { return _impl; }
    [[nodiscard]] auto impl() && noexcept { return std::move(_impl); }

    /**
     * @brief Get the runtime directory where backends are located.
     * @return Path to the runtime directory.
     */
    [[nodiscard]] const luisa::filesystem::path &runtime_directory() const noexcept;

    /**
     * @brief Get the data directory for caches and persistent data.
     * @return Path to the data directory.
     */
    [[nodiscard]] const luisa::filesystem::path &data_directory() const noexcept;

    /**
     * @brief Create a subdirectory under the runtime directory.
     * @param folder_name Name of the subdirectory to create.
     * @return Path to the created subdirectory.
     */
    [[nodiscard]] const luisa::filesystem::path &create_runtime_subdir(luisa::string_view folder_name) const noexcept;

    /**
     * @brief Create a backend device.
     * @param backend_name Name of the backend (e.g., "cuda", "dx", "metal", "cpu").
     * @param settings Optional device configuration settings.
     * @param enable_validation Whether to enable the validation layer.
     * @return A Device object representing the created backend device.
     * 
     * Logic: This method loads the dynamic library `luisa-backend-<name>`,
     * instantiates the device interface, and returns a high-level Device wrapper.
     */
    [[nodiscard]] Device create_device(luisa::string_view backend_name,
                                       const DeviceConfig *settings,
                                       bool enable_validation) noexcept;

    /**
     * @brief Create a backend device with default validation mode.
     * @param backend_name Name of the backend.
     * @param settings Optional device configuration settings.
     * @return A Device object.
     * 
     * Logic: Validation mode is determined by the environment variable `LUISA_ENABLE_VALIDATION`.
     */
    [[nodiscard]] Device create_device(luisa::string_view backend_name,
                                       const DeviceConfig *settings = nullptr) noexcept;

    /**
     * @brief Get a list of installed backends.
     * @return A span of backend names detected in the runtime directory.
     */
    [[nodiscard]] luisa::span<const luisa::string> installed_backends() const noexcept;

    /**
     * @brief Create a device using the first available backend.
     * @return A Device object.
     * @note Panics if no backends are found.
     */
    [[nodiscard]] Device create_default_device() noexcept;

    /**
     * @brief Get the names of physical devices available for a backend.
     * @param backend_name Name of the backend.
     * @return A vector of device description strings.
     */
    [[nodiscard]] luisa::vector<luisa::string> backend_device_names(luisa::string_view backend_name) const noexcept;

    /**
     * @brief Load a backend dynamic module.
     * @param backend_name Name of the backend.
     * @return Reference to the loaded DynamicModule.
     */
    [[nodiscard]] const DynamicModule &load_backend(luisa::string_view backend_name) const noexcept;
};

}// namespace luisa::compute
