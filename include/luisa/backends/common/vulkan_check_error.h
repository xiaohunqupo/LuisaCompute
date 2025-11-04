//
// Created by mike on 11/4/25.
//

#pragma once

#include <luisa/core/magic_enum.h>
#include <luisa/core/logging.h>

#define LUISA_CHECK_VULKAN(x)                                            \
    do {                                                                 \
        auto ret = x;                                                    \
        if (ret != VK_SUCCESS) [[unlikely]] {                            \
            if (ret > 0 || ret == VK_ERROR_OUT_OF_DATE_KHR) [[likely]] { \
                LUISA_WARNING_WITH_LOCATION(                             \
                    "Vulkan call `" #x "` returned {} (code = {}).",     \
                    ::luisa::to_string(ret), luisa::to_underlying(ret)); \
            } else [[unlikely]] {                                        \
                LUISA_ERROR_WITH_LOCATION(                               \
                    "Vulkan call `" #x "` failed: {} (code = {}).",      \
                    ::luisa::to_string(ret), luisa::to_underlying(ret)); \
            }                                                            \
        }                                                                \
    } while (false)
