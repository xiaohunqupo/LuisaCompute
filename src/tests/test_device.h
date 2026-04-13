// Common device creation helper for standalone tests.
//
// Usage from main():
//   auto [context, device] = luisa::test::create_device(argc, argv);
//
// Usage from Boost.UT test:
//   auto [context, device] = luisa::test::create_device_from_ut();
//
// Both parse the backend name from argv[1] and print usage on missing arg.

#pragma once

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>

namespace luisa::test {

struct DeviceContext {
    compute::Context context;
    compute::Device device;
};

/// Create a device from main(argc, argv).
/// Exits with code 1 if no backend argument is provided.
[[nodiscard]] inline DeviceContext create_device(int argc, char *argv[]) {
    compute::Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    compute::Device device = context.create_device(argv[1]);
    return {std::move(context), std::move(device)};
}

/// Create a device from Boost.UT's stored argc/argv
/// (available when linking with ut.hpp).
/// Returns std::nullopt if no backend argument is provided.
[[nodiscard]] inline std::optional<DeviceContext> create_device_from_ut() {
    auto argc = boost::ut::detail::cfg::largc;
    auto argv = boost::ut::detail::cfg::largv;
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        return std::nullopt;
    }
    compute::Context context{argv[0]};
    compute::Device device = context.create_device(argv[1]);
    return DeviceContext{std::move(context), std::move(device)};
}

}// namespace luisa::test
