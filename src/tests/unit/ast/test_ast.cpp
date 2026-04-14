// Test for AST (Abstract Syntax Tree) type system and attribute handling.
// This test verifies the creation of buffer types with custom attributes.

#include <luisa/luisa-compute.h>
#include <exception>
#include "ut/ut.hpp"
#include "test_device.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int test_ast(Device &device) {
    static_cast<void>(device);
    // Enable verbose logging for debugging
    luisa::log_level_verbose();

    // Create a list of custom attributes for the buffer type
    luisa::vector<Attribute> attris;
    attris.emplace_back("attr0", "attr1");

    // Create a buffer type with float elements and custom attributes
    auto t = Type::buffer(Type::of<float>(), attris);

    // Print the type description to verify attribute handling
    LUISA_INFO("{}", t->description());
    return 0;
}

static inline const auto reg = [] {
    "ast"_test = [] {
        auto dc = luisa::test::create_device_from_ut();
        if (!dc) return;
        auto &device = dc->device;
        try {
            test_ast(device);
            expect(true);
        } catch (const std::exception &e) {
            expect(false) << e.what();
        } catch (...) {
            expect(false) << "unknown exception";
        }
    };
    return 0;
}();

int main() {}
