#include <luisa/vstl/string_builder.h>

namespace vstd {
StringBuilder::~StringBuilder() = default;
StringBuilder &StringBuilder::append(vstd::string_view str) {
    vstd::push_back_all(vec, str.data(), str.size());
    return *this;
}
StringBuilder &StringBuilder::append(char str) {
    vec.push_back(str);
    return *this;
}
StringBuilder &StringBuilder::append(vstd::string const &str) {
    vstd::push_back_all(vec, str.data(), str.size());
    return *this;
}
StringBuilder::StringBuilder() = default;

LUISA_VSTL_API void to_string(float val, StringBuilder &builder) noexcept {
    const size_t len = snprintf(nullptr, 0, "%a", val);
    auto lastLen = builder.size();
    builder.push_back(len + 1);
    auto iter = builder.end() - 1;
    *iter = 0;
    snprintf(builder.data() + lastLen, len + 1, "%a", val);
    *iter = 'f';
}

LUISA_VSTL_API void to_string(double val, StringBuilder &builder) noexcept {
    const size_t len = snprintf(nullptr, 0, "%a", val);
    auto lastLen = builder.size();
    builder.push_back(len + 1);
    auto iter = builder.end() - 1;
    *iter = 0;
    snprintf(builder.data() + lastLen, len + 1, "%a", val);
    builder.erase(builder.end() - 1);
}

}// namespace vstd

