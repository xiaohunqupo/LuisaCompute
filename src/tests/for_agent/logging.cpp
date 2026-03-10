// Test for logging functionality.
// This test verifies that the logging system works correctly with different
// log levels, formats, and macros.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../common/doctest.h"
#include <luisa/core/logging.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

// Simple test to verify logging doesn't crash and produces output
TEST_CASE("logging basic functionality") {
    // Test basic log functions with simple messages
    luisa::log_verbose("Verbose message from test");
    luisa::log_info("Info message from test");
    luisa::log_warning("Warning message from test");
    
    // Test formatted log messages
    luisa::log_verbose("Verbose with args: {}, {}", 42, 3.14);
    luisa::log_info("Info with args: {}, {}", "test_string", true);
    luisa::log_warning("Warning with args: {}, {}", 'a', 123u);
    
    // Test flush
    luisa::log_flush();
}

TEST_CASE("logging macros") {
    // Test all logging macros
    LUISA_VERBOSE("Macro verbose message");
    LUISA_INFO("Macro info message");
    LUISA_WARNING("Macro warning message");
    
    // Test macros with format arguments
    LUISA_VERBOSE("Verbose macro: {}, {}", 1, 2);
    LUISA_INFO("Info macro: {}, {}, {}", "a", "b", "c");
    LUISA_WARNING("Warning macro: value = {}", 3.14159);
}

TEST_CASE("logging with location macros") {
    // Test location macros
    LUISA_VERBOSE_WITH_LOCATION("Verbose with location: {}", 100);
    LUISA_INFO_WITH_LOCATION("Info with location: {}", 200);
    LUISA_WARNING_WITH_LOCATION("Warning with location: {}", 300);
}

TEST_CASE("logging log level changes") {
    // Save current log level (we can't query it, but we can test setting doesn't crash)
    
    // Test all log level functions
    luisa::log_level_verbose();
    LUISA_VERBOSE("This verbose should be visible after log_level_verbose()");
    
    luisa::log_level_info();
    LUISA_VERBOSE("This verbose should be hidden after log_level_info()");
    LUISA_INFO("This info should be visible after log_level_info()");
    
    luisa::log_level_warning();
    LUISA_VERBOSE("This verbose should be hidden after log_level_warning()");
    LUISA_INFO("This info should be hidden after log_level_warning()");
    LUISA_WARNING("This warning should be visible after log_level_warning()");
    
    luisa::log_level_error();
    LUISA_VERBOSE("This verbose should be hidden after log_level_error()");
    LUISA_INFO("This info should be hidden after log_level_error()");
    LUISA_WARNING("This warning should be hidden after log_level_error()");
    
    // Reset to verbose for other tests
    luisa::log_level_verbose();
    luisa::log_flush();
}

TEST_CASE("logging complex format strings") {
    // Test various format specifiers
    LUISA_INFO("Integer: {}", -123456);
    LUISA_INFO("Unsigned: {}", 123456u);
    LUISA_INFO("Float: {}", 3.14159265f);
    LUISA_INFO("Double: {}", 2.718281828459045);
    LUISA_INFO("String: {}", "test_string");
    LUISA_INFO("Boolean: {}", true);
    LUISA_INFO("Pointer: {}", static_cast<void*>(nullptr));
    LUISA_INFO("Hex: {:x}", 255);
    LUISA_INFO("Binary: {:b}", 170);
    LUISA_INFO("Scientific: {:e}", 12345.6789);
    LUISA_INFO("Fixed: {:.2f}", 3.14159);
}

TEST_CASE("logging empty and special messages") {
    // Test empty message (via format)
    LUISA_INFO("");
    
    // Test message with special characters
    LUISA_INFO("Special chars: \t\n\r!@#$%^&*()_+-=[]{}|;':\",./<>?");
    
    // Test message with braces
    LUISA_INFO("Braces: {{escaped}}, {{{{nested}}}}");
    
    // Test long message
    luisa::string long_message;
    for (int i = 0; i < 100; ++i) {
        long_message += "This is a long message repeated multiple times. ";
    }
    LUISA_INFO("Long message: {}", long_message);
}
