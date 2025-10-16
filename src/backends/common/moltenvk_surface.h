//
// Created by mike on 10/16/25.
//

#pragma once

#ifdef LUISA_PLATFORM_APPLE

#include <cstdint>

namespace luisa::compute {
void *cocoa_window_content_view(uint64_t window_handle) noexcept;
}// namespace luisa::compute

#endif
