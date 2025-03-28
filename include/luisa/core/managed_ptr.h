#pragma once

#include <atomic>
#include <luisa/core/stl/memory.h>

namespace luisa {

class ManagedObject;

template<typename T>
    requires std::derived_from<T, ManagedObject>
class ManagedPtr;

class ManagedObject {

private:
    std::atomic<size_t> _ref_count{1ul};

public:
    ManagedObject() noexcept = default;
    virtual ~ManagedObject() = default;
    ManagedObject(ManagedObject &&) = delete;
    ManagedObject(const ManagedObject &) = delete;
    ManagedObject &operator=(ManagedObject &&) = delete;
    ManagedObject &operator=(const ManagedObject &) = delete;

private:
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    friend class ManagedPtr;

    ManagedObject *retain() noexcept;
    void release() noexcept;
};

template<typename T>
    requires std::derived_from<T, ManagedObject>
class ManagedPtr {

private:
    T *_object{nullptr};
    [[nodiscard]] static T *_retained_if_not_null(T *o) noexcept {
        return o == nullptr ? nullptr : static_cast<T *>(o->retain());
    }
    explicit ManagedPtr(T *object) noexcept
        : _object{object} {}

public:
    ManagedPtr() noexcept = default;
    ~ManagedPtr() noexcept { reset(); }
    ManagedPtr(ManagedPtr &&other) noexcept
        : _object{other.transfer()} {}
    ManagedPtr(const ManagedPtr &other) noexcept
        : _object{_retained_if_not_null(other.get())} {}
    ManagedPtr &operator=(ManagedPtr &&other) noexcept {
        if (&other != this) {
            reset(other.transfer());
        }
        return *this;
    }
    ManagedPtr &operator=(const ManagedPtr &other) noexcept {
        if (&other != this) {
            reset(_retained_if_not_null(other.get()));
        }
        return *this;
    }

public:
    [[nodiscard]] T *get() noexcept { return _object; }
    [[nodiscard]] T *get() const noexcept { return _object; }
    [[nodiscard]] T *operator->() noexcept { return get(); }
    [[nodiscard]] T *operator->() const noexcept { return get(); }
    [[nodiscard]] T &operator*() noexcept { return *get(); }
    [[nodiscard]] T &operator*() const noexcept { return *get(); }

public:
    [[nodiscard]] T *transfer() noexcept {
        return std::exchange(_object, nullptr);
    }
    void reset(T *new_object = nullptr) noexcept {
        if (auto old_object = std::exchange(_object, new_object)) {
            old_object->release();
        }
    }
};

template<typename T, typename... Args>
    requires std::derived_from<T, ManagedObject>
[[nodiscard]] ManagedPtr<T> make_managed(Args &&...args) noexcept {
    auto o = luisa::new_with_allocator<T>(std::forward<Args>(args)...);
    assert(std::addressof(*o) == std::addressof(*static_cast<ManagedObject *>(o)) &&
           "ManagedObject should be the first base class of its derived classes.");
    ManagedPtr<T> ret;
    ret.reset(o);
    return ret;
}

}// namespace luisa
