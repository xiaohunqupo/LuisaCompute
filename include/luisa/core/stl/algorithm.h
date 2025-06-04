#pragma once

#ifdef LUISA_USE_SYSTEM_STL
#include <algorithm>
#else
#include <EASTL/algorithm.h>
#endif

#include <luisa/core/stl/pdqsort.h>

namespace luisa {

using std::swap;

#ifdef LUISA_USE_SYSTEM_STL
using std::transform;
using std::binary_search;
#else
using eastl::transform;
using eastl::binary_search;
#endif

template<typename Begin, typename End>
void sort(Begin &&begin, End &&end) noexcept {
    pdqsort(std::forward<Begin>(begin),
            std::forward<End>(end));
}

template<typename Begin, typename End, typename Compare>
void sort(Begin &&begin, End &&end, Compare &&comp) noexcept {
    pdqsort(std::forward<Begin>(begin),
            std::forward<End>(end),
            std::forward<Compare>(comp));
}

}// namespace luisa
