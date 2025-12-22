#pragma once

#include <marl/conditionvariable.h>
#include <marl/containers.h>
#include <marl/export.h>
#include <marl/memory.h>

#include <luisa/core/stl/type_traits.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/concepts.h>
#include <luisa/core/stl/memory.h>

namespace luisa::fiber {

// Future is a synchronization primitive used to block until a signal is raised.
template<typename T>
class Future {
public:
    Future(marl::Allocator *allocator = marl::Allocator::Default);
    template<typename... Args>
        requires(luisa::is_constructible_v<T, Args && ...>)
    void signal(Args &&...) const;

    // clear() clears the signaled state.
    void clear() const;

    // wait() blocks until the event is signaled.
    // If the event was constructed with the Auto Mode, then only one
    // call to wait() will unblock before returning, upon which the signalled
    // state will be automatically cleared.
    [[nodiscard]] T &wait() const;

    // test() returns true if the event is signaled, otherwise false.
    // If the event is signalled and was constructed with the Auto Mode
    // then the signalled state will be automatically cleared upon returning.
    [[nodiscard]] bool test() const;

    // isSignalled() returns true if the event is signaled, otherwise false.
    // Unlike test() the signal is not automatically cleared when the event was
    // constructed with the Auto Mode.
    // Note: No lock is held after bool() returns, so the event state may
    // immediately change after returning. Use with caution.
    [[nodiscard]] bool isSignalled() const;
    Future(Future const &) = default;
    Future(Future &&) = default;
    Future &operator=(Future &&rhs) {
        if (std::addressof(rhs) == this) [[unlikely]]
            return *this;
        std::destroy_at(this);
        std::construct_at(this, std::move(rhs));
        return *this;
    }
    Future &operator=(Future const &rhs) {
        if (std::addressof(rhs) == this) [[unlikely]]
            return *this;
        std::destroy_at(this);
        std::construct_at(this, rhs);
        return *this;
    }

private:
    struct Shared {
        [[nodiscard]] Shared(marl::Allocator *allocator);
        template<typename... Args>
        void signal(Args &&...args);
        T &wait();

        marl::mutex mutex;
        marl::ConditionVariable cv;
        luisa::optional<T> result;
    };
    using SharedPtr = decltype(std::declval<marl::Allocator>().make_shared<Shared>(std::declval<marl::Allocator *>()));
    const SharedPtr shared;
};
template<typename T>
inline Future<T>::Shared::Shared(marl::Allocator *allocator) : cv(allocator) {}

template<typename T>
inline T &Future<T>::Shared::wait() {
    marl::lock lock(mutex);
    cv.wait(lock, [&] { return static_cast<bool>(result); });
    return *result;
}

template<typename T>
inline Future<T>::Future(marl::Allocator *allocator /* = marl::Allocator::Default */)
    : shared(allocator->make_shared<Shared>(allocator)) {}

template<typename T>
template<typename... Args>
    requires(luisa::is_constructible_v<T, Args && ...>)
inline void Future<T>::signal(Args &&...args) const {
    shared->signal(std::forward<Args>(args)...);
}

template<typename T>
template<typename... Args>
inline void Future<T>::Shared::signal(Args &&...args) {
    marl::lock lock(mutex);
    result.reset();
    result.emplace(std::forward<Args>(args)...);
    cv.notify_all();
}

template<typename T>
inline void Future<T>::clear() const {
    marl::lock lock(shared->mutex);
    shared->result.reset();
}

template<typename T>
inline T &Future<T>::wait() const {
    return shared->wait();
}

template<typename T>
inline bool Future<T>::test() const {
    marl::lock lock(shared->mutex);
    if (!shared->result) {
        return false;
    }
    return true;
}

template<typename T>
inline bool Future<T>::isSignalled() const {
    marl::lock lock(shared->mutex);
    return shared->result.has_value();
}

}// namespace luisa::fiber
