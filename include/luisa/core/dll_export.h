#pragma once

#ifdef __cplusplus
#define LUISA_EXTERN_C extern "C"
#define LUISA_NOEXCEPT noexcept
#else
#define LUISA_EXTERN_C
#define LUISA_NOEXCEPT
#endif

#ifdef _MSC_VER
#define LUISA_FORCE_INLINE inline
#define LUISA_NEVER_INLINE __declspec(noinline)
#define LUISA_EXPORT_API LUISA_EXTERN_C __declspec(dllexport)
#define LUISA_IMPORT_API LUISA_EXTERN_C __declspec(dllimport)
#else
#define LUISA_FORCE_INLINE __attribute__((always_inline, hot)) inline
#define LUISA_NEVER_INLINE __attribute__((noinline))
#define LUISA_EXPORT_API LUISA_EXTERN_C __attribute__((visibility("default")))
#define LUISA_IMPORT_API LUISA_EXTERN_C
#endif

#ifdef _MSC_VER

#ifdef LUISA_CORE_EXPORT_DLL
#define LUISA_CORE_API __declspec(dllexport)
#else
#define LUISA_CORE_API __declspec(dllimport)
#endif

#ifdef LUISA_VSTL_STATIC_LIB
#define LUISA_VSTL_API
#else
#ifdef LUISA_VSTL_EXPORT_DLL
#define LUISA_VSTL_API __declspec(dllexport)
#else
#define LUISA_VSTL_API __declspec(dllimport)
#endif
#endif

#ifdef LUISA_GUI_EXPORT_DLL
#define LUISA_GUI_API __declspec(dllexport)
#else
#define LUISA_GUI_API __declspec(dllimport)
#endif

#ifdef LUISA_AST_EXPORT_DLL
#define LUISA_AST_API __declspec(dllexport)
#else
#define LUISA_AST_API __declspec(dllimport)
#endif

#ifdef LUISA_RUNTIME_EXPORT_DLL
#define LUISA_RUNTIME_API __declspec(dllexport)
#else
#define LUISA_RUNTIME_API __declspec(dllimport)
#endif

#ifdef LUISA_DSL_STATIC_LIB
#define LUISA_DSL_API
#else
#ifdef LUISA_DSL_EXPORT_DLL
#define LUISA_DSL_API __declspec(dllexport)
#else
#define LUISA_DSL_API __declspec(dllimport)
#endif
#endif

#ifdef LUISA_TENSOR_STATIC_LIB
#define LUISA_TENSOR_API
#else
#ifdef LUISA_TENSOR_EXPORT_DLL
#define LUISA_TENSOR_API __declspec(dllexport)
#else
#define LUISA_TENSOR_API __declspec(dllimport)
#endif
#endif

#ifdef LUISA_OSL_EXPORT_DLL
#define LUISA_OSL_API __declspec(dllexport)
#else
#define LUISA_OSL_API __declspec(dllimport)
#endif

#ifdef LUISA_IR_EXPORT_DLL
#define LUISA_IR_API __declspec(dllexport)
#else
#define LUISA_IR_API __declspec(dllimport)
#endif

#ifdef LUISA_SERDE_LIB_EXPORT_DLL
#define LUISA_SERDE_LIB_API __declspec(dllexport)
#else
#define LUISA_SERDE_LIB_API __declspec(dllimport)
#endif

#ifdef LUISA_REMOTE_EXPORT_DLL
#define LUISA_REMOTE_API __declspec(dllexport)
#else
#define LUISA_REMOTE_API __declspec(dllimport)
#endif

#ifdef LUISA_BACKEND_EXPORT_DLL
#define LUISA_BACKEND_API __declspec(dllexport)
#else
#define LUISA_BACKEND_API __declspec(dllimport)
#endif

#ifdef LUISA_CLANGCXX_EXPORT_DLL
#define LUISA_CLANGCXX_API __declspec(dllexport)
#else
#define LUISA_CLANGCXX_API __declspec(dllimport)
#endif

#ifdef LUISA_XIR_EXPORT_DLL
#define LUISA_XIR_API __declspec(dllexport)
#else
#define LUISA_XIR_API __declspec(dllimport)
#endif

#else
#define LUISA_CORE_API
#define LUISA_VSTL_API
#define LUISA_AST_API
#define LUISA_RUNTIME_API
#define LUISA_DSL_API
#define LUISA_TENSOR_API
#define LUISA_OSL_API
#define LUISA_IR_API
#define LUISA_SERDE_LIB_API
#define LUISA_REMOTE_API
#define LUISA_GUI_API
#define LUISA_BACKEND_API
#define LUISA_CLANGCXX_API
#define LUISA_XIR_API
#endif

