#pragma once

#include <new>
#include <utility>
#include <type_traits>
#include <EASTL/internal/type_pod.h>

namespace luisa {

template<class T, class... Args>
constexpr bool is_constructible_v = eastl::is_constructible_v<T, Args...>;
}// namespace luisa
