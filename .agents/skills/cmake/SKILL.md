---
name: cmake
---

# CMake Build Guide for LuisaCompute

## Requirements

- CMake 3.26+
- Ninja (recommended)
- C++20 compiler (MSVC, Clang, or GCC)

## Quick Start

```bash
# Configure with Ninja
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release

# Build
cmake --build build

# Install
cmake --install build --prefix dist
```

## Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | - | `Release` or `Debug` |
| `LUISA_COMPUTE_ENABLE_DSL` | ON | C++ DSL support |
| `LUISA_COMPUTE_ENABLE_CUDA` | ON | CUDA backend |
| `LUISA_COMPUTE_ENABLE_METAL` | ON | Metal backend (macOS) |
| `LUISA_COMPUTE_ENABLE_DX` | ON | DirectX backend (Windows) |
| `LUISA_COMPUTE_ENABLE_VULKAN` | ON | Vulkan backend |
| `LUISA_COMPUTE_ENABLE_CPU` | ON | CPU backend |
| `LUISA_COMPUTE_ENABLE_REMOTE` | ON | Remote backend |
| `LUISA_COMPUTE_ENABLE_FALLBACK` | ON | Fallback backend |
| `LUISA_COMPUTE_ENABLE_GUI` | ON | GUI support |
| `LUISA_COMPUTE_ENABLE_UNITY_BUILD` | OFF | Unity build (faster compile) |
| `LUISA_COMPUTE_ENABLE_SANITIZERS` | OFF | Address/UB sanitizers |
| `LUISA_COMPUTE_USE_SYSTEM_LIBS` | OFF | Prefer system libraries |

## Platform Notes

### Linux
```bash
# With specific compiler
export CC=clang-20 CXX=clang++-20
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release
```

### macOS
```bash
# Use Homebrew LLVM
export PATH="$PATH:/opt/homebrew/opt/llvm/bin"
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export SDKROOT=$(xcrun --show-sdk-path)
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release
```

### Windows

**Using bootstrap (Recommended):**
The bootstrap script automatically detects and sets up the MSVC environment:
```python
import bootstrap
bootstrap.prepare_msvc_environment()
```

**Manual CMake:**
Requires running from VS Developer Command Prompt:
```cmd
# From VS Developer Command Prompt
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release
```

## CI Example (Minimal)

```bash
cmake -S . -B build -G Ninja \
  -D LUISA_COMPUTE_ENABLE_RUST=OFF \
  -D LUISA_COMPUTE_ENABLE_REMOTE=OFF \
  -D LUISA_COMPUTE_ENABLE_CPU=OFF \
  -D CMAKE_BUILD_TYPE=Release

cmake --build build
```


## Build System Architecture

### Target Naming Conventions

| Prefix | Example | Purpose |
|--------|---------|---------|
| `luisa-compute-<module>` | `luisa-compute-core` | Internal library targets |
| `luisa-compute-backend-<name>` | `luisa-compute-backend-cuda` | Backend plugin targets (output as `luisa-backend-<name>`) |
| `luisa-compute-ext-<name>` | `luisa-compute-ext-spdlog` | Third-party extension targets |
| `luisa::compute` | Alias | Interface target linking all core modules |

### Module Hierarchy

```
luisa-compute-include (INTERFACE header-only)
    ↓
luisa-compute-ext (INTERFACE third-party deps)
    ↓
luisa-compute-core (SHARED)
    ↓
luisa-compute-ast (SHARED) → luisa-compute-xir (SHARED)
    ↓
luisa-compute-runtime (SHARED)
    ↓
luisa-compute-dsl, luisa-compute-gui, luisa-compute-ir
    ↓
luisa-compute-backends (INTERFACE aggregator)
```

### Custom CMake Functions

#### `luisa_compute_install(target)`

Installs a target with consistent destination paths:
```cmake
luisa_compute_install(core SOURCES ${LUISA_COMPUTE_CORE_SOURCES})
```

#### `luisa_compute_add_backend(name)`

Creates a backend plugin MODULE target:
```cmake
luisa_compute_add_backend(cuda SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})
```

- Creates `luisa-compute-backend-${name}` MODULE
- Links to `ast`, `runtime`, `gui`, optionally `dsl`
- Sets output name to `luisa-backend-${name}`
- Installs to `${CMAKE_INSTALL_BINDIR}` (bin, not lib)

#### `luisa_compute_add_executable(name)`

Creates executable linked to `luisa::compute`:
```cmake
luisa_compute_add_executable(my_app)
```

#### `luisa_compute_test_suite(name)` / `luisa_compute_add_test(name)`

Test creation helpers:
```cmake
luisa_compute_test_suite(feat)  # globs next/test/feat/**.cpp
luisa_compute_add_test(my_test) # adds to test_main executable
```

### Backend Plugin Build Pattern

Backends are built as `MODULE` (shared libraries) loaded at runtime:
```cmake
luisa_compute_add_backend(cuda SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})
```

Key properties:
- Output renamed to `luisa-backend-<name>` (without `compute-`)
- Installed to `bin/` not `lib/` because they are runtime plugins
- Support for builtin device libraries via `luisa_embed_device_lib`

### Rust Integration

**File**: `src/rust/CMakeLists.txt`

1. Custom command invokes `cargo build`
2. Profile: `dev` (Debug) or `release` (Release)
3. CMake targets:
   - `luisa-compute-rust-meta` (INTERFACE): Static Rust libs
   - `luisa_compute_backend_impl` (INTERFACE): Shared Rust backend

### Third-Party Extension Pattern

Each extension in `src/ext/` follows:
```cmake
if (LUISA_COMPUTE_USE_SYSTEM_<LIB>)
    find_package(<LIB> REQUIRED)
    target_link_libraries(luisa-compute-ext INTERFACE <target>)
    target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_<LIB>=1)
else ()
    add_subdirectory(<lib>)
    target_link_libraries(luisa-compute-ext INTERFACE <target>)
    luisa_compute_install_extension(<target> ...)
endif ()
```

### Output Directories

```
${CMAKE_BINARY_DIR}/bin  → Runtime outputs (DLLs, executables)
${CMAKE_BINARY_DIR}/lib  → Archive outputs (static libs, PDBs)
```

### RPATH Configuration

- **macOS**: `@loader_path`, `@loader_path/../bin`, `@loader_path/../lib`
- **Linux**: `$ORIGIN`, `$ORIGIN/../bin`, `$ORIGIN/../lib`
