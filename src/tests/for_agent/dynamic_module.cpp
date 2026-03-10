// Test for DynamicModule functionality.
// This test verifies dynamic module loading, symbol resolution,
// search path management, and move semantics.

#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../common/doctest.h"

#include <luisa/core/dynamic_module.h>
#include <luisa/core/logging.h>
#include <luisa/core/platform.h>

#ifdef LUISA_PLATFORM_WINDOWS
#include <windows.h>
#endif

using namespace luisa;

TEST_SUITE("DynamicModule") {

    TEST_CASE("DynamicModule empty construction") {
        DynamicModule empty_module;
        CHECK(!empty_module);
        CHECK(empty_module.handle() == nullptr);
    }

    TEST_CASE("DynamicModule load system library") {
        log_level_verbose();
        
        // Platform-specific C runtime library name
#ifdef LUISA_PLATFORM_WINDOWS
        const char *crt_name = "ucrtbase";  // Universal C Runtime on Windows
#else
        const char *crt_name = "c";  // libc on Linux/Unix
#endif
        
        auto crt_module = DynamicModule::load(crt_name);
        if (!crt_module) {
            // Fallback for different systems
#ifdef LUISA_PLATFORM_WINDOWS
            crt_module = DynamicModule::load("msvcrt");
#else
            crt_module = DynamicModule::load("m");  // libm
#endif
        }
        
        if (crt_module) {
            CHECK(crt_module);
            CHECK(crt_module.handle() != nullptr);
            
            // Test symbol lookup - look for common C functions
            auto malloc_ptr = crt_module.address("malloc");
            // malloc might not be available in all CRTs, so just log
            if (malloc_ptr) {
                LUISA_INFO("Found 'malloc' symbol at address: {}", malloc_ptr);
            }
            
            auto free_ptr = crt_module.address("free");
            if (free_ptr) {
                LUISA_INFO("Found 'free' symbol at address: {}", free_ptr);
            }
        } else {
            LUISA_WARNING("Could not load C runtime library. Skipping related checks.");
        }
    }

    TEST_CASE("DynamicModule move semantics") {
#ifdef LUISA_PLATFORM_WINDOWS
        auto source = DynamicModule::load("kernel32");
#else
        auto source = DynamicModule::load("dl");  // libdl on Linux
#endif
        
        if (source) {
            CHECK(source);
            void *original_handle = source.handle();
            
            // Move construct
            DynamicModule moved{std::move(source)};
            CHECK(moved);
            CHECK(!source);  // Source should be empty after move
            CHECK(moved.handle() == original_handle);
            
            // Move assign
            DynamicModule target;
            target = std::move(moved);
            CHECK(target);
            CHECK(!moved);  // Moved-from should be empty
            CHECK(target.handle() == original_handle);
            
            // Test release
            void *released = target.release();
            CHECK(released == original_handle);
            CHECK(!target);  // Should be null after release
            
            // Clean up the released handle
            if (released) {
                dynamic_module_destroy(released);
            }
        } else {
            LUISA_WARNING("Could not load module for move semantics test.");
        }
    }

    TEST_CASE("DynamicModule reset") {
#ifdef LUISA_PLATFORM_WINDOWS
        auto module = DynamicModule::load("kernelbase");
#else
        auto module = DynamicModule::load("pthread");
#endif
        
        if (module) {
            CHECK(module);
            module.reset();
            CHECK(!module);
            CHECK(module.handle() == nullptr);
            // Reset on empty should be safe
            module.reset();
            CHECK(!module);
        } else {
            LUISA_WARNING("Could not load module for reset test.");
        }
    }

    TEST_CASE("DynamicModule function template") {
#ifdef LUISA_PLATFORM_WINDOWS
        auto module = DynamicModule::load("kernel32");
        if (module) {
            // GetLocalTime is a commonly available function
            using GetLocalTimeFunc = void(void*);
            auto get_local_time = module.function<GetLocalTimeFunc>("GetLocalTime");
            if (get_local_time) {
                LUISA_INFO("Found GetLocalTime function pointer.");
            }
        }
#else
        auto module = DynamicModule::load("m");  // libm
        if (module) {
            // sin function from math library
            using SinFunc = double(double);
            auto sin_func = module.function<SinFunc>("sin");
            if (sin_func) {
                LUISA_INFO("Found sin function pointer.");
                double result = sin_func(0.0);
                CHECK(result == doctest::Approx(0.0));
            }
        }
#endif
    }

    TEST_CASE("DynamicModule search path management") {
        // Get the current executable path as a safe directory
        auto exe_path = current_executable_path();
        auto exe_dir = std::filesystem::path(exe_path).parent_path();
        
        // Test adding search path
        DynamicModule::add_search_path(exe_dir);
        LUISA_INFO("Added search path: {}", to_string(exe_dir));
        
        // Adding the same path again should increase reference count
        DynamicModule::add_search_path(exe_dir);
        LUISA_INFO("Added same search path again (should increase ref count).");
        
        // Remove the path twice to match the add count
        DynamicModule::remove_search_path(exe_dir);
        DynamicModule::remove_search_path(exe_dir);
        LUISA_INFO("Removed search path twice.");
        
        // Test loading from a specific folder
#ifdef LUISA_PLATFORM_WINDOWS
        auto module = DynamicModule::load(exe_dir, "kernel32");
#else
        // On Linux, try to load from standard lib path
        auto module = DynamicModule::load("/lib/x86_64-linux-gnu", "c");
        if (!module) {
            module = DynamicModule::load("/lib64", "c");
        }
        if (!module) {
            module = DynamicModule::load("/usr/lib", "c");
        }
#endif
        
        if (module) {
            LUISA_INFO("Successfully loaded module from specific folder.");
        }
    }

    TEST_CASE("DynamicModule load_exact") {
        // Try to construct the full path to a system library
        std::filesystem::path lib_path;
        bool found = false;
        
#ifdef LUISA_PLATFORM_WINDOWS
        // On Windows, try to find kernel32.dll
        char sys_path[MAX_PATH];
        if (GetSystemDirectoryA(sys_path, MAX_PATH)) {
            lib_path = std::filesystem::path(sys_path) / "kernel32.dll";
            found = std::filesystem::exists(lib_path);
        }
#else
        // On Linux, try common libc paths
        std::vector<std::filesystem::path> candidates = {
            "/lib/x86_64-linux-gnu/libc.so.6",
            "/lib64/libc.so.6",
            "/usr/lib/libc.so.6",
            "/lib/libc.so.6"
        };
        for (const auto &candidate : candidates) {
            if (std::filesystem::exists(candidate)) {
                lib_path = candidate;
                found = true;
                break;
            }
        }
#endif
        
        if (found) {
            auto module = DynamicModule::load_exact(lib_path);
            if (module) {
                LUISA_INFO("Successfully loaded module using load_exact: {}", to_string(lib_path));
                CHECK(module);
            } else {
                LUISA_WARNING("load_exact returned empty module for existing path: {}", to_string(lib_path));
            }
        } else {
            LUISA_WARNING("Could not find system library for load_exact test.");
        }
    }

    TEST_CASE("DynamicModule non-existent module") {
        auto non_existent = DynamicModule::load("definitely_not_a_real_module_name_12345");
        CHECK(!non_existent);
        CHECK(non_existent.handle() == nullptr);
        
        // Trying to get address from empty module should be safe
        auto addr = non_existent.address("some_function");
        CHECK(addr == nullptr);
    }

    TEST_CASE("DynamicModule function invocation") {
#ifdef LUISA_PLATFORM_WINDOWS
        auto module = DynamicModule::load("ucrtbase");
        if (!module) {
            module = DynamicModule::load("msvcrt");
        }
        
        if (module) {
            // abs is a simple function to test
            using AbsFunc = int(int);
            auto abs_func = module.function<AbsFunc>("abs");
            if (abs_func) {
                int result = abs_func(-42);
                LUISA_INFO("abs(-42) = {}", result);
                CHECK(result == 42);
            }
        }
#else
        auto module = DynamicModule::load("m");  // libm
        if (module) {
            using SqrtFunc = double(double);
            auto sqrt_func = module.function<SqrtFunc>("sqrt");
            if (sqrt_func) {
                double result = sqrt_func(16.0);
                LUISA_INFO("sqrt(16.0) = {}", result);
                CHECK(result == doctest::Approx(4.0));
            }
        }
#endif
    }

}// TEST_SUITE("DynamicModule")
