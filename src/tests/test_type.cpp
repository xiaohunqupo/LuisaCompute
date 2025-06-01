#include <string>
#include <fstream>
#include <memory>
#include <variant>
#include <atomic>
#include <iostream>

#include <luisa/luisa-compute.h>

struct S1 {
    float x;
};

struct S2 {
    float x;
    float y;
};

struct S3 {
    float x;
    float y;
    float z;
};

struct S4 {
    float x;
    float y;
    float z;
    float w;
};

struct Test {

    std::string s;
    int a;

    template<typename Archive>
    void serialize(Archive &&ar) noexcept {
        ar(s, a);
    }
};

using namespace luisa;
using namespace luisa::compute;

struct alignas(16) AA {
    float4 x;
    float ba[16];
    float a;
};

struct BB {
    AA a;
    float b;
    float3x3 m;
};

LUISA_STRUCT_REFLECT(AA, x, ba, a)
LUISA_STRUCT_REFLECT(BB, a, b, m)

struct Interface : public concepts::Noncopyable {
    Interface() noexcept = default;
    Interface(Interface &&) noexcept = default;
    Interface &operator=(Interface &&) noexcept = default;
    ~Interface() noexcept = default;
};

template<typename T>
    requires concepts::container<T>
void foo(T &&) noexcept {}

struct Impl : public Interface {};

class Something : public luisa::Managed<Something> {};

struct SomeValue : Something {
    int value;
    explicit SomeValue(int v = -1) noexcept : value{v} {}
    ~SomeValue() noexcept override { LUISA_INFO("SomeValue destroyed with value: {}", value); }
};

struct SomeNode : public luisa::ManagedIntrusiveNode<SomeNode, SomeValue> {
    using Super::Super;
};

struct SomeForwardNode : public luisa::ManagedIntrusiveForwardNode<SomeForwardNode, SomeValue> {
    using Super::Super;
};

std::string_view tag_name(Type::Tag tag) noexcept {
    using namespace std::string_view_literals;
    if (tag == Type::Tag::BOOL) { return "bool"sv; }
    if (tag == Type::Tag::FLOAT32) { return "float"sv; }
    if (tag == Type::Tag::INT32) { return "int"sv; }
    if (tag == Type::Tag::UINT32) { return "uint"sv; }
    if (tag == Type::Tag::VECTOR) { return "vector"sv; }
    if (tag == Type::Tag::MATRIX) { return "matrix"sv; }
    if (tag == Type::Tag::ARRAY) { return "array"sv; }
    if (tag == Type::Tag::STRUCTURE) { return "struct"sv; }
    return "unknown"sv;
}

template<int max_level = -1>
void print(const Type *info, int level = 0) {

    std::string indent_string;
    for (auto i = 0; i < level; i++) { indent_string.append("  "); }
    if (max_level >= 0 && level > max_level) {
        std::cout << indent_string << info->description() << "\n";
        return;
    }

    std::cout << indent_string << tag_name(info->tag()) << ": {\n"
              << indent_string << "  size:        " << info->size() << "\n"
              << indent_string << "  alignment:   " << info->alignment() << "\n"
              << indent_string << "  hash:        " << info->hash() << "\n"
              << indent_string << "  description: " << info->description() << "\n";

    if (info->is_structure()) {
        std::cout << indent_string << "  members:\n";
        for (auto m : info->members()) { print<max_level>(m, level + 2); }
    } else if (info->is_vector() || info->is_array() || info->is_matrix()) {
        std::cout << indent_string << "  dimension:   " << info->dimension() << "\n";
        std::cout << indent_string << "  element:\n";
        print<max_level>(info->element(), level + 2);
    }
    std::cout << indent_string << "}\n";
}

int main() {

    using namespace luisa;
    log_level_verbose();

    LUISA_VERBOSE("verbose...");
    LUISA_VERBOSE_WITH_LOCATION("verbose with {}...", "location");
    LUISA_INFO("info...");
    LUISA_INFO_WITH_LOCATION("info with location...");
    LUISA_WARNING("warning...");
    LUISA_WARNING_WITH_LOCATION("warning with location...");

    LUISA_INFO("size = {}, alignment = {}", sizeof(AA), alignof(AA));
    LUISA_INFO("size = {}, alignment = {}", sizeof(BB), alignof(BB));
    LUISA_INFO("trivially destructible: {}", std::is_trivially_destructible_v<Impl>);
    print(Type::from("array<array<vector<float,3>,5>,9>"));

    LUISA_INFO("{}", Type::of<std::array<float, 5>>()->description());

    int aa[1024];
    print(Type::of(aa));

    BB bb;
    print(Type::of(bb));

    static_assert(alignof(float3) == 16);

    auto u = make_float2(1.0f, 2.0f);
    auto v = make_float3(1.0f, 2.0f, 3.0f);
    auto w = make_float3(u, 1.0f);

    auto vv = v + w;
    auto bvv = v == w;
    static_assert(std::is_same_v<decltype(bvv), bool3>);
    v += w;
    v *= 10.0f;

    v = 2.0f * v;
    auto ff = v[2];
    ff = 1.0f;
    auto tt = make_float2(v);

    print(Type::of<float3x3>());

    foo<std::initializer_list<int>>({1, 2, 3, 4});

    auto [m, n] = std::array{1, 2};

    auto sth = luisa::make_managed<Something>();
    sth = sth;
    sth = std::move(sth);
    {
        auto another = sth;
        luisa::ManagedPtr<const Something> good = std::move(another);
        LUISA_ASSERT(nullptr == another);
        auto gg = good.get();
        LUISA_ASSERT(gg == sth);
        auto ggg = gg->lock();
        auto more = good->lock();
        LUISA_ASSERT(sth == more);
        more = std::move(ggg);
        LUISA_ASSERT(more != nullptr);
        good = more;
        LUISA_ASSERT(good);
        another = sth;
    }

    {
        luisa::ManagedPtr<const luisa::detail::ManagedObject> bad = std::move(sth);
        auto worse = std::move(bad).into<Something>();
    }

    luisa::unordered_set<luisa::ManagedPtr<Something>> set;

    LUISA_INFO("Begin managed intrusive list test...");
    {
        luisa::ManagedIntrusiveList<SomeNode> list;
        auto n1 = list.push_front(make_managed<SomeNode>(1));// [1]
        auto n2 = list.push_back(make_managed<SomeNode>(2)); // [1, 2]
        auto n3 = list.push_front(make_managed<SomeNode>(3));// [3, 1, 2]
        auto n4 = list.push_back(make_managed<SomeNode>(4)); // [3, 1, 2, 4]
        {
            auto rm_n2 = n2->remove_self();// [3, 1, 4]
            LUISA_ASSERT(!rm_n2->is_linked(), "Node should be unlinked after removal.");
        }
        auto n5 = n3->insert_after_self(make_managed<SomeNode>(5)); // [3, 5, 1, 4]
        auto n6 = n5->insert_before_self(make_managed<SomeNode>(6));// [3, 6, 5, 1, 4]
        {
            auto rm_n3 = n3->remove_self();// [6, 5, 1, 4]
            LUISA_ASSERT(!rm_n3->is_linked(), "Node should be unlinked after removal.");
            auto rm_n4 = n4->remove_self();// [6, 5, 1]
            LUISA_ASSERT(!rm_n4->is_linked(), "Node should be unlinked after removal.");
        }
        for (auto node : list) {
            LUISA_INFO("Node value: {}", node->value);
        }
        for (auto iter = list.crbegin(); iter != list.crend(); ++iter) {
            LUISA_INFO("Reverse Node value: {}", (*iter)->value);
        }
    }
    LUISA_INFO("End managed intrusive list test...");

    LUISA_INFO("Begin managed intrusive forward list test...");
    {
        luisa::ManagedIntrusiveForwardList<SomeForwardNode> list;
        auto n1 = list.push_front(make_managed<SomeForwardNode>(1)->lock_into<SomeForwardNode>());// [1]
        auto n2 = list.push_front(make_managed<SomeForwardNode>(2));// [2, 1]
        auto n3 = list.push_front(make_managed<SomeForwardNode>(3));// [3, 2, 1]
        auto n4 = list.push_front(make_managed<SomeForwardNode>(4));// [4, 3, 2, 1]
        {
            auto rm_n2 = n2->remove_self();// [4, 3, 1]
            LUISA_ASSERT(!rm_n2->is_linked(), "Node should be unlinked after removal.");
        }
        auto n5 = list.push_front(make_managed<SomeForwardNode>(5));// [5, 4, 3, 1]
        auto n6 = list.push_front(make_managed<SomeForwardNode>(6));// [6, 5, 4, 3, 1]
        {
            auto rm_n1 = n1->remove_self();// [6, 5, 4, 3]
            LUISA_ASSERT(!rm_n1->is_linked(), "Node should be unlinked after removal.");
            auto rm_n6 = n6->remove_self();// [5, 4, 3]
            LUISA_ASSERT(!rm_n6->is_linked(), "Node should be unlinked after removal.");
        }
        auto n7 = list.push_front(make_managed<SomeForwardNode>(7));// [7, 5, 4, 3]
        auto n8 = list.push_front(make_managed<SomeForwardNode>(8));// [8, 7, 5, 4, 3]
        {
            auto rm_n5 = n5->remove_self();// [8, 7, 4, 3]
            LUISA_ASSERT(!rm_n5->is_linked(), "Node should be unlinked after removal.");
        }
        for (auto node : list) {
            LUISA_INFO("Node value: {}", node->value);
        }
    }
    LUISA_INFO("End managed intrusive forward list test...");
}
