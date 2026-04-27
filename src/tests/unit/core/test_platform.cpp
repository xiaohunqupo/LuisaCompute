// Test for platform utilities.
// Covers: current_executable_path, env_separator, aligned_alloc/free,
//         pagesize, dynamic_module_prefix/extension, dynamic_module_name.

#include "ut/ut.hpp"

#include <cstring>
#include <luisa/core/platform.h>
#include <luisa/core/logging.h>

using namespace boost::ut;
using namespace boost::ut::literals;

// ---- current_executable_path ----

static inline const auto reg_exe_path = [] {
    "current_executable_path"_test = [] {
        auto path = luisa::current_executable_path();
        expect(!path.empty()) << "executable path should not be empty";
        // path should contain the test executable name
        LUISA_INFO("Executable path: {}", path);
    };
    return 0;
}();

// ---- env_separator ----

static inline const auto reg_env_separator = [] {
    "env_separator"_test = [] {
        char sep = luisa::env_separator();
#ifdef _WIN32
        expect(sep == ';') << "env separator on Windows should be ';'";
#else
        expect(sep == ':') << "env separator on POSIX should be ':'";
#endif
    };
    return 0;
}();

// ---- dynamic_module_name ----

static inline const auto reg_dynamic_module_name = [] {
    "dynamic_module_name_composition"_test = [] {
        auto name = luisa::dynamic_module_name("test_module");
        expect(!name.empty()) << "dynamic_module_name should return non-empty string";
#ifdef _WIN32
        expect(static_cast<bool>(name == "test_module.dll"))
            << "Windows: expected 'test_module.dll' but got '" << name.c_str() << "'";
#elif defined(__APPLE__)
        expect(static_cast<bool>(name == "libtest_module.so"))
            << "macOS: expected 'libtest_module.so' but got '" << name.c_str() << "'";
#else
        expect(static_cast<bool>(name == "libtest_module.so"))
            << "Linux: expected 'libtest_module.so' but got '" << name.c_str() << "'";
#endif
    };
    return 0;
}();

// ---- aligned_alloc / aligned_free ----

static inline const auto reg_aligned_alloc_basic = [] {
    "aligned_alloc_basic"_test = [] {
        void *p = luisa::aligned_alloc(16u, 128u);
        expect(static_cast<bool>(p != nullptr)) << "aligned_alloc should return non-null";
        // verify alignment
        auto addr = reinterpret_cast<uintptr_t>(p);
        expect((addr % 16u) == 0u) << "pointer should be aligned to 16 bytes";

        // write/read to verify usable memory
        std::memset(p, 0xAB, 128u);
        expect(static_cast<unsigned char *>(p)[0] == 0xABu);
        expect(static_cast<unsigned char *>(p)[127] == 0xABu);

        luisa::aligned_free(p);
    };
    return 0;
}();

static inline const auto reg_aligned_alloc_various_alignments = [] {
    "aligned_alloc_various_alignments"_test = [] {
        for (size_t align : {8u, 16u, 32u, 64u, 128u, 256u}) {
            void *p = luisa::aligned_alloc(align, 256u);
            expect(static_cast<bool>(p != nullptr));
            auto addr = reinterpret_cast<uintptr_t>(p);
            expect((addr % align) == 0u)
                << "pointer should be aligned to " << align << " bytes";
            luisa::aligned_free(p);
        }
    };
    return 0;
}();

static inline const auto reg_aligned_free_null = [] {
    "aligned_free_null"_test = [] {
        // Freeing null should be safe
        luisa::aligned_free(nullptr);
        expect(true);
    };
    return 0;
}();

// ---- pagesize ----

static inline const auto reg_pagesize = [] {
    "pagesize"_test = [] {
        auto ps = luisa::pagesize();
        expect(ps > 0u) << "pagesize must be positive";
        // pagesize should be a power of 2
        expect((ps & (ps - 1u)) == 0u) << "pagesize should be a power of 2";
        // Common page sizes are 4K, 16K, 64K
        expect(ps >= 4096u) << "pagesize should be at least 4096";
        LUISA_INFO("Page size: {} bytes", ps);
    };
    return 0;
}();

// ---- cpu_name ----

static inline const auto reg_cpu_name = [] {
    "cpu_name"_test = [] {
        auto name = luisa::cpu_name();
        expect(!name.empty()) << "cpu_name should not be empty";
        LUISA_INFO("CPU name: {}", name);
    };
    return 0;
}();

// ---- backtrace ----

static inline const auto reg_backtrace = [] {
    "backtrace"_test = [] {
        auto trace = luisa::backtrace();
        // Should have at least one frame (this function)
        expect(!trace.empty()) << "backtrace should return at least one frame";
        // Each frame should have a non-zero address
        for (const auto &item : trace) {
            expect(item.address != 0u);
        }
    };
    return 0;
}();

int main() {}
