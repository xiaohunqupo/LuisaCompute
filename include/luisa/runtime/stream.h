#pragma once

#include <utility>

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/stream_event.h>
#include <luisa/runtime/command_list.h>

namespace luisa::compute {

/**
 * @brief Command queue for asynchronous device execution.
 *
 * The Stream class represents a logically FIFO command queue on the device.
 * It supports submitting commands (like shader dispatches or memory transfers)
 * using the fluent `operator<<` syntax.
 * 
 * Logic: When a command is piped into a Stream, it is buffered into an internal
 * CommandList. The commands are dispatched to the backend when:
 * 1. The statement ends (RAII-based flush via Delegate).
 * 2. An explicit `commit()` or `synchronize()` is piped.
 */
class LUISA_RUNTIME_API Stream final : public Resource {

public:
    struct Commit {};
    struct Synchronize {};

    /**
     * @brief Proxy object for buffering commands in a single statement.
     * 
     * Logic: Delegate collects multiple commands and dispatches them together
     * when the proxy is destroyed at the end of the statement. This improves
     * backend efficiency by reducing the number of dispatch calls.
     */
    class LUISA_RUNTIME_API Delegate {

    private:
        Stream *_stream;
        CommandList _command_list;

    private:
        void _commit() noexcept;

    private:
        friend class Stream;
        explicit Delegate(Stream *s) noexcept;
        Delegate(Delegate &&) noexcept;

    public:
        ~Delegate() noexcept;
        Delegate(const Delegate &) noexcept = delete;
        Delegate &operator=(Delegate &&) noexcept = delete;
        Delegate &operator=(const Delegate &) noexcept = delete;

        /**
         * @brief Add a command to the buffered list.
         * @param cmd Unique pointer to the command.
         * @return Moving reference to self for chaining.
         */
        Delegate operator<<(luisa::unique_ptr<Command> &&cmd) && noexcept;

        /**
         * @brief Add a host-side callback to the buffered list.
         * @param f Move-only function to execute on the host after preceding commands finish.
         * @return Moving reference to self.
         */
        Delegate operator<<(luisa::move_only_function<void()> &&f) && noexcept;

        template<typename T>
            requires std::is_rvalue_reference_v<T &&> && is_stream_event_v<T>
        Stream &operator<<(T &&t) && noexcept {
            _commit();
            luisa::invoke(std::forward<T>(t), _stream->device(), _stream->handle());
            return *_stream;
        }

        /**
         * @brief Explicitly commit the buffered commands to the backend.
         */
        Stream &operator<<(CommandList::Commit &&commit) && noexcept;

        /**
         * @brief Commit buffered commands and wait for completion.
         */
        Stream &operator<<(Synchronize &&) && noexcept;

        /**
         * @brief Explicitly commit the buffered commands.
         */
        Stream &operator<<(Commit &&) && noexcept;

        // compound commands
        template<typename... T>
        decltype(auto) operator<<(std::tuple<T...> args) && noexcept {
            auto encode = [&]<size_t... i>(std::index_sequence<i...>) noexcept -> decltype(auto) {
                return (std::move(*this) << ... << std::move(std::get<i>(args)));
            };
            return encode(std::index_sequence_for<T...>{});
        }
    };

private:
    friend class Device;
    friend class DStorageExt;
    StreamTag _stream_tag{};

private:
    explicit Stream(DeviceInterface *device, StreamTag stream_tag) noexcept;
    explicit Stream(DeviceInterface *device, StreamTag stream_tag, const ResourceCreationInfo &stream_handle) noexcept;
    void _dispatch(CommandList &&command_buffer) noexcept;
    void _synchronize() noexcept;

public:
    Stream() noexcept = default;
    ~Stream() noexcept override;
    Stream(Stream &&) noexcept = default;
    Stream(Stream const &) noexcept = delete;
    Stream &operator=(Stream &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Stream &operator=(Stream const &) noexcept = delete;
    using Resource::operator bool;
    using Resource::release;

    /**
     * @brief Start a command submission sequence.
     * @param cmd The first command.
     * @return A Delegate proxy for buffering subsequent commands.
     */
    Delegate operator<<(luisa::unique_ptr<Command> &&cmd) noexcept;

    /**
     * @brief Start a command submission with a host callback.
     */
    Delegate operator<<(luisa::move_only_function<void()> &&f) noexcept;

    template<typename T>
        requires std::is_rvalue_reference_v<T &&> && is_stream_event_v<T>
    Stream &operator<<(T &&t) noexcept {
        luisa::invoke(std::forward<T>(t), device(), handle());
        return *this;
    }

    /**
     * @brief Submit an external command list.
     */
    Stream &operator<<(CommandList::Commit &&commit) noexcept;

    /**
     * @brief Synchronize the stream.
     */
    Stream &operator<<(Synchronize &&) noexcept;

    /**
     * @brief Block the host thread until all commands in this stream finish.
     */
    void synchronize() noexcept { _synchronize(); }

    /// @return The tag (Compute, Graphics, etc.) of this stream.
    [[nodiscard]] auto stream_tag() const noexcept { return _stream_tag; }

    // compound commands
    template<typename... T>
    decltype(auto) operator<<(std::tuple<T...> &&args) noexcept {
        // FIXME: Delegate{this} << without a temporary definition may boom GCC
        Delegate delegate{this};
        return std::move(delegate) << std::move(args);
    }

    using LogCallback = DeviceInterface::StreamLogCallback;

    /**
     * @brief Set a callback for backend logging.
     * @param callback Function to receive log messages from the backend.
     */
    void set_log_callback(const LogCallback &callback) noexcept;
};

[[nodiscard]] constexpr auto commit() noexcept { return Stream::Commit{}; }
[[nodiscard]] constexpr auto synchronize() noexcept { return Stream::Synchronize{}; }

}// namespace luisa::compute

