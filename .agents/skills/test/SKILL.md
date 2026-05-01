# LuisaCompute Test Guide

## Overview
Tests are standalone binaries using [Boost.UT](https://github.com/boost-ext/ut) (`src/tests/ut/ut.hpp`). Built when `lc_enable_tests=true` (default).

## Layout
| Directory | Content | Needs Device |
|-----------|---------|--------------|
| `unit/core/` | basic_types, traits, io, math, logging | No |
| `unit/ast/` | AST construction, manual AST, builtin kernels | Yes |
| `unit/dsl/` | DSL sugar, structs, callables, buffers | Yes |
| `unit/runtime/` | buffer, texture, stream, warp, atomics | Yes |
| `unit/xir/` | XIR builder, translators (gated by `lc_enable_xir`) | Yes |
| `integration/runtime/` | multi-stream, swapchain, rtx, raster | Yes |
| `integration/ir/` | autodiff, AST<->IR (gated by `lc_enable_ir`) | Yes |

## Adding a Test

In `src/tests/xmake.lua`:
```lua
test_proj("test_my_feature", "unit/runtime/test_my_feature.cpp")
```
Optional args:
- `gui_dep=true` — skips if `lc_enable_gui=false`, defines `LUISA_ENABLE_GUI`
- `callable` — extra target config, e.g. `function() add_deps("lc-osl") end`

## Test File Template
```cpp
#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/runtime/device.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_my_feature(Device &device) {
    // ...
    expect(true);
}

static inline const auto reg = [] {
    "my_feature"_test = [] {
        auto dc = luisa::test::create_device_from_ut();
        if (!dc) return;
        test_my_feature(dc->device);
    };
    return 0;
}();

int main() {}
```

## Device Helpers (`test_device.h`)
- `luisa::test::create_device(argc, argv)` — from `main()`; exits on missing backend
- `luisa::test::create_device_from_ut()` — from UT registration; returns `std::nullopt` on missing backend

Backend is passed as the first positional argument: `cuda`, `dx`, `cpu`, `metal`.

## Assertions
Use Boost.UT macros:
- `expect(condition)`
- `expect(condition) << "message"`
- `expect(bool_expr) << "msg"`

For float comparison use helpers like `float_eq(a, b, eps)`.

## Running Tests
```bash
# build all tests
xmake -g tests

# run one test with a backend
xmake run test_dsl cuda
./bin/test_dsl dx

# filter tests by name/pattern (Boost.UT CLI)
./bin/test_basic_types "vector*"
```

## Key Dependencies
Tests link against `lc-runtime`, `lc-dsl`, `lc-vstl`, `stb-image`, and optionally `lc-gui`. The dummy backend `lc-backends-dummy` is added as a non-linking dependency.
