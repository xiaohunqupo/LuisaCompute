#pragma once

#include <atomic>
#include <luisa/core/stl/memory.h>

namespace luisa {

namespace detail {
class ManagedObject;

class ManagedObject {

private:
    std::atomic<uint32_t> _ref_count;
    uint32_t _managed_id;

public:
    explicit ManagedObject(uint32_t m_id = 0u) noexcept : _ref_count{1u}, _managed_id{m_id} {}
    virtual ~ManagedObject() = default;
    ManagedObject(ManagedObject &&) = delete;
    ManagedObject(const ManagedObject &) = delete;
    ManagedObject &operator=(ManagedObject &&) = delete;
    ManagedObject &operator=(const ManagedObject &) = delete;

protected:
    ManagedObject *do_retain() noexcept;
    void do_release() noexcept;

public:
    [[nodiscard]] auto managed_id() const noexcept { return _managed_id; }
    void set_managed_id(uint32_t m_id) noexcept { _managed_id = m_id; }
};

}// namespace detail

template<typename T>
class ManagedPtr {

    static_assert(std::derived_from<T, detail::ManagedObject>);

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

template<typename T, typename Base = detail::ManagedObject>
    requires std::derived_from<Base, detail::ManagedObject>
class Managed : public Base {
public:
    using derived_type = T;
    using base_type = Base;
    using super_type = Managed;

public:
    using base_type::base_type;

private:
    friend ManagedPtr<T>;
    derived_type *retain() noexcept { return static_cast<derived_type *>(this->do_retain()); }
    void release() noexcept { this->do_release(); }
};

template<typename T, typename... Args>
    requires std::derived_from<T, detail::ManagedObject>
[[nodiscard]] ManagedPtr<T> make_managed(Args &&...args) noexcept {
    auto o = luisa::new_with_allocator<T>(std::forward<Args>(args)...);
    assert(std::addressof(*o) == std::addressof(*static_cast<detail::ManagedObject *>(o)) &&
           "ManagedObject should be the first base class of its derived classes.");
    ManagedPtr<T> ret;
    ret.reset(o);
    return ret;
}

}// namespace luisa
