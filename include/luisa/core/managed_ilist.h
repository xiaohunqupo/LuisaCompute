#pragma once

#include <luisa/core/stl/iterator.h>
#include <luisa/core/concepts.h>
#include <luisa/core/managed_ptr.h>

namespace luisa {

template<typename Node, typename SentinelNode = Node>
class ManagedIntrusiveList : public concepts::Noncopyable {

    static_assert(std::derived_from<SentinelNode, Node>);

private:
    template<typename T, typename Advance>
    class IteratorBase {

    private:
        friend ManagedIntrusiveList;
        T *_current = nullptr;
        explicit IteratorBase(T *current) noexcept : _current{current} {}

    public:
        [[nodiscard]] auto operator*() const noexcept -> T * {
            assert(_current != nullptr && !_current->is_sentinel() && "Dereferencing a null or sentinel node.");
            return _current;
        }
        [[nodiscard]] auto operator->() const noexcept -> T * {
            assert(_current != nullptr && !_current->is_sentinel() && "Dereferencing a null or sentinel node.");
            return _current;
        }
        [[nodiscard]] auto operator==(luisa::default_sentinel_t) const noexcept -> bool {
            return _current->is_sentinel();
        }
        IteratorBase &operator++() noexcept {
            Advance::advance(_current);
            return *this;
        }
    };

    struct ForwardAdvance {
        template<typename U>
        void static advance(U &current) noexcept { current = current->next(); }
    };
    struct BackwardAdvance {
        template<typename U>
        void static advance(U &current) noexcept { current = current->prev(); }
    };

private:
    ManagedPtr<Node> _head_sentinel{nullptr};
    Node *_tail_sentinel{nullptr};

public:
    template<typename... Args>
    explicit ManagedIntrusiveList(Args... args) noexcept {
        _head_sentinel = make_managed<SentinelNode>(args...);
        _head_sentinel->_prev = nullptr;
        _head_sentinel->_next = make_managed<SentinelNode>(args...);
        _tail_sentinel = _head_sentinel->_next.get();
    }
    ~ManagedIntrusiveList() noexcept {
        while (!empty()) {
            static_cast<void>(pop_front());
        }
    }

public:// accessors
    [[nodiscard]] auto head_sentinel() noexcept -> Node * { return _head_sentinel.get(); }
    [[nodiscard]] auto head_sentinel() const noexcept -> const Node * { return _head_sentinel.get(); }
    [[nodiscard]] auto tail_sentinel() noexcept -> Node * { return _tail_sentinel; }
    [[nodiscard]] auto tail_sentinel() const noexcept -> const Node * { return _tail_sentinel; }
    [[nodiscard]] auto empty() const noexcept -> bool { return _head_sentinel->next() == _tail_sentinel; }
    [[nodiscard]] auto front() noexcept -> Node * {
        assert(!empty() && "Cannot access front of an empty list.");
        return head_sentinel()->next();
    }
    [[nodiscard]] auto back() noexcept -> Node * {
        assert(!empty() && "Cannot access back of an empty list.");
        return tail_sentinel()->prev();
    }
    [[nodiscard]] auto front() const noexcept -> const Node * { return const_cast<ManagedIntrusiveList *>(this)->front(); }
    [[nodiscard]] auto back() const noexcept -> const Node * { return const_cast<ManagedIntrusiveList *>(this)->back(); }

public:// iterators
    using Iterator = IteratorBase<Node, ForwardAdvance>;
    using ConstIterator = IteratorBase<const Node, ForwardAdvance>;
    using ReverseIterator = IteratorBase<Node, BackwardAdvance>;
    using ConstReverseIterator = IteratorBase<const Node, BackwardAdvance>;
    [[nodiscard]] auto begin() noexcept -> Iterator { return Iterator{front()}; }
    [[nodiscard]] auto begin() const noexcept -> ConstIterator { return ConstIterator{front()}; }
    [[nodiscard]] auto end() const noexcept -> luisa::default_sentinel_t { return luisa::default_sentinel; }
    [[nodiscard]] auto rbegin() noexcept -> ReverseIterator { return ReverseIterator{back()}; }
    [[nodiscard]] auto rbegin() const noexcept -> ConstReverseIterator { return ConstReverseIterator{back()}; }
    [[nodiscard]] auto rend() const noexcept -> luisa::default_sentinel_t { return luisa::default_sentinel; }
    [[nodiscard]] auto cbegin() const noexcept { return this->begin(); }
    [[nodiscard]] auto cend() const noexcept { return this->end(); }
    [[nodiscard]] auto crbegin() const noexcept { return this->rbegin(); }
    [[nodiscard]] auto crend() const noexcept { return this->rend(); }

public:// modifiers
    void push_front(ManagedPtr<Node> node) noexcept { head_sentinel()->insert_after_self(std::move(node)); }
    void push_back(ManagedPtr<Node> node) noexcept { tail_sentinel()->insert_before_self(std::move(node)); }
    auto pop_front() noexcept -> ManagedPtr<Node> { return front()->remove_self(); }
    auto pop_back() noexcept -> ManagedPtr<Node> { return back()->remove_self(); }
};

template<typename T, typename Base = detail::ManagedObject>
class ManagedIntrusiveNode : public Base {

    template<typename, typename>
    friend class ManagedIntrusiveList;

public:
    using Super = ManagedIntrusiveNode;
    static_assert(std::derived_from<Base, detail::ManagedObject>);

protected:
    using Base::Base;

private:
    T *_prev{nullptr};
    ManagedPtr<T> _next{nullptr};

public:
    [[nodiscard]] auto prev() noexcept -> T * { return _prev; }
    [[nodiscard]] auto prev() const noexcept -> const T * { return _prev; }
    [[nodiscard]] auto next() noexcept -> T * { return _next.get(); }
    [[nodiscard]] auto next() const noexcept -> const T * { return _next.get(); }
    [[nodiscard]] auto is_linked() const noexcept -> bool { return _prev != nullptr && _next != nullptr; }
    [[nodiscard]] auto is_head_sentinel() const noexcept -> bool { return _prev == nullptr; }
    [[nodiscard]] auto is_tail_sentinel() const noexcept -> bool { return _next == nullptr; }
    [[nodiscard]] auto is_sentinel() const noexcept -> bool { return is_head_sentinel() || is_tail_sentinel(); }

public:
    virtual auto remove_self() noexcept -> ManagedPtr<T> {
        if (!is_linked()) [[unlikely]] { return nullptr; }
        assert(!is_sentinel() && "Cannot remove a sentinel node.");
        assert(_prev->next() == this && "Node is not linked correctly.");
        assert(_next->prev() == this && "Node is not linked correctly.");
        // process the previous node first before we nullify the pointers
        _next->_prev = _prev;
        // we should hold self to prevent being a dangling pointer
        auto self = std::exchange(_prev->_next, std::move(_next));
        assert(_next == nullptr && "Next pointer should be null after removal.");
        // nullify the pointers
        _prev = nullptr;
        return self;
    }
    virtual void insert_before_self(ManagedPtr<T> node) noexcept {
        assert(!node->is_linked() && "Inserting a linked node into a list.");
        assert(!is_head_sentinel() && "Inserting before a head sentinel.");
        // get a pointer to the node being inserted before moving it
        auto p_node = node.get();
        p_node->_prev = _prev;
        auto self = std::exchange(_prev->_next, std::move(node));
        p_node->_next = std::move(self);
        _prev = p_node;
    }
    virtual void insert_after_self(ManagedPtr<T> node) noexcept {
        assert(!node->is_linked() && "Inserting a linked node into a list.");
        assert(!is_tail_sentinel() && "Inserting after a tail sentinel.");
        // insert after self <==> insert before the next node
        _next->insert_before_self(std::move(node));
    }
};

template<typename Node>
class ManagedIntrusiveForwardList {

private:
    template<typename T>
    class IteratorBase {

    private:
        friend ManagedIntrusiveForwardList;
        T *_current = nullptr;
        explicit IteratorBase(T *current) noexcept : _current{current} {}

    public:
        [[nodiscard]] auto operator*() const noexcept -> T * {
            assert(_current != nullptr && "Dereferencing a null node.");
            return _current;
        }
        [[nodiscard]] auto operator->() const noexcept -> T * {
            assert(_current != nullptr && "Dereferencing a null node.");
            return _current;
        }
        [[nodiscard]] auto operator==(luisa::default_sentinel_t) const noexcept -> bool {
            return _current == nullptr;
        }
        IteratorBase &operator++() noexcept {
            _current = _current->next();
            return *this;
        }
    };

private:
    ManagedPtr<Node> _head{nullptr};

public:
    ManagedIntrusiveForwardList() noexcept = default;
    ManagedIntrusiveForwardList(ManagedIntrusiveForwardList &&) noexcept = delete;
    ManagedIntrusiveForwardList(const ManagedIntrusiveForwardList &) noexcept = delete;
    ManagedIntrusiveForwardList &operator=(ManagedIntrusiveForwardList &&) noexcept = delete;
    ManagedIntrusiveForwardList &operator=(const ManagedIntrusiveForwardList &) noexcept = delete;
    ~ManagedIntrusiveForwardList() noexcept {
        while (!empty()) {
            static_cast<void>(pop_front());
        }
    }

public:// accessors
    [[nodiscard]] auto empty() const noexcept -> bool { return _head == nullptr; }
    [[nodiscard]] auto front() noexcept -> Node * {
        assert(!empty() && "Cannot access front of an empty list.");
        return _head;
    }
    [[nodiscard]] auto front() const noexcept -> const Node * {
        return const_cast<ManagedIntrusiveForwardList *>(this)->front();
    }

public:// iterators
    using Iterator = IteratorBase<Node>;
    using ConstIterator = IteratorBase<const Node>;
    [[nodiscard]] auto begin() noexcept -> Iterator { return Iterator{front()}; }
    [[nodiscard]] auto begin() const noexcept -> ConstIterator { return ConstIterator{front()}; }
    [[nodiscard]] auto end() const noexcept -> luisa::default_sentinel_t { return luisa::default_sentinel; }
    [[nodiscard]] auto cbegin() const noexcept { return this->begin(); }
    [[nodiscard]] auto cend() const noexcept { return this->end(); }

public:// modifiers
    void push_front(ManagedPtr<Node> node) noexcept {
        assert(node->_next == nullptr && node->_prev_next == nullptr && "Node is already linked.");
        auto new_head = node.get();
        if (auto old_head = std::exchange(_head, std::move(node))) {
            old_head->_prev_next = &new_head->_next;
            new_head->_next = std::move(old_head);
        }
        new_head->_prev_next = &_head;
    }
    auto pop_front() noexcept -> ManagedPtr<Node> { return front()->remove_self(); }
};

template<typename T, typename Base = detail::ManagedObject>
class ManagedIntrusiveForwardNode : public Base {

    template<typename, typename>
    friend class ManagedIntrusiveForwardList;

public:
    using Super = ManagedIntrusiveForwardNode;
    static_assert(std::derived_from<Base, detail::ManagedObject>);

protected:
    using Base::Base;

private:
    ManagedPtr<T> _next{nullptr};      // pointer to the next node
    ManagedPtr<T> *_prev_next{nullptr};// pointer to the next pointer of the previous node

public:
    [[nodiscard]] auto next() noexcept -> T * { return _next.get(); }
    [[nodiscard]] auto next() const noexcept -> const T * { return _next.get(); }
    [[nodiscard]] auto is_linked() const noexcept -> bool { return _prev_next != nullptr; }
    virtual auto remove_self() noexcept -> ManagedPtr<T> {
        if (!is_linked()) [[unlikely]] { return nullptr; }
        if (_next != nullptr) { _next->_prev_next = _prev_next; }
        auto self = std::exchange(*_prev_next, std::move(_next));
        assert(self == this && "The node being removed is not the same as the current node.");
        assert(self->_next == nullptr && "Next pointer should be null after removal.");
        _prev_next = nullptr;// nullify the pointer to the next pointer of the previous node
        return self;
    }
};

}// namespace luisa
