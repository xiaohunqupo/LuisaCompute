#pragma once

#include <luisa/core/dll_export.h>

#define VSTL_ABORT() std::abort()

#if defined(UNICODE) && !defined(VSTL_UNICODE)
#define VSTL_UNICODE
#endif

#define VSTL_EXPORT_C LUISA_EXPORT_API

#if (defined(DEBUG) || defined(_DEBUG) || !defined(NDEBUG)) && !defined(VSTL_DEBUG)
#define VSTL_DEBUG
#endif

#define VENGINE_C_FUNC_COMMON
#define VENGINE_EXIT std::abort()

#include <cstdint>

using uint = uint32_t;
using uint64 = uint64_t;
using int64 = int64_t;
using int32 = int32_t;
