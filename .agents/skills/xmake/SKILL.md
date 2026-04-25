---
name: xmake
---

# XMake Build System

This project uses XMake as its primary build system.

## Requirements

- XMake 3.0.6+
- CUDA Toolkit (optional, for CUDA backend)
- Vulkan SDK (optional, for Vulkan backend)
- LLVM 20 (optional, for clang toolchain)
- Rust (for some dependencies)

## Quick Start

```bash
# Configure and build with default options
xmake f -m debug -c
xmake build

# Build with specific configuration
xmake f -m release -c
xmake build
```

## Update compile_commands.json
```bash
xmake project -k compile_commands --lsp=clangd .vscode
```

## Configuration Commands

```bash
# Linux with GCC
xmake f -p linux -a x86_64 --toolchain=gcc -m release -c

# Linux with Clang
xmake f -p linux -a x86_64 --toolchain=clang -m release -c

# Windows with MSVC
xmake f -p windows -a x64 --toolchain=msvc -m release -c

# Windows with Clang-CL
xmake f -p windows -a x64 --toolchain=clang-cl -m release -c

# Windows with LLVM
xmake f -p windows -a x64 --toolchain=llvm --sdk="C:/Program Files/LLVM" -m release -c
```

## Common Flags

| Flag | Description |
|------|-------------|
| `-c` | Clean configuration cache |
| `--check` | Check flags before building |
| `-m <mode>` | Build mode: release, debug, releasedbg |
| `-p <plat>` | Platform: linux, windows, macosx |
| `-a <arch>` | Architecture: x86_64, x64, arm64 |

## Project Options

### Backends

| Option | Default | Description |
|--------|---------|-------------|
| `lc_cuda_backend` | true | Enable NVIDIA CUDA backend |
| `lc_vk_backend` | true | Enable Vulkan backend |
| `lc_dx_backend` | true | Enable DirectX-12 backend |
| `lc_metal_backend` | true | Enable Metal backend |
| `lc_fallback_backend` | false | Enable fallback CPU backend |
| `lc_toy_c_backend` | false | Enable toy C backend for testing |

### Features

| Option | Default | Description |
|--------|---------|-------------|
| `lc_enable_dsl` | true | Enable C++ DSL module |
| `lc_enable_gui` | true | Enable GUI module |
| `lc_enable_imgui` | true | Enable ImGui integration |
| `lc_enable_tests` | true | Enable tests module |
| `lc_enable_py` | true | Enable Python bindings |
| `lc_enable_osl` | true | Enable OpenShadingLanguage support |

### Build Optimization

| Option | Default | Description |
|--------|---------|-------------|
| `lc_enable_unity_build` | true | Enable unity (jumbo) build |
| `lc_enable_simd` | true | Enable SSE and SSE2 SIMD |
| `lc_use_lto` | false | Enable Link Time Optimization |
| `lc_enable_mimalloc` | true | Use mimalloc as default allocator |
| `lc_enable_custom_malloc` | false | Use custom memory allocator |

### Advanced Options

| Option | Default | Description |
|--------|---------|-------------|
| `lc_cxx_standard` | cxx20 | C++ standard version |
| `lc_rtti` | false | Enable C++ RTTI |
| `lc_enable_xir` | false | Enable XIR (IR) support |
| `lc_dx_cuda_interop` | false | Enable DirectX-CUDA interop |
| `lc_vk_cuda_interop` | false | Enable Vulkan-CUDA interop |
| `lc_cuda_ext_lcub` | false | Enable CUDA CUB extension |
| `lc_sdk_dir` | false | SDK download directory |
| `lc_bin_dir` | bin | Binary output directory |

## Usage Examples

```bash
# Full build with all backends and tests
xmake f -p linux -m release \
  --lc_cuda_backend=true \
  --lc_vk_backend=true \
  --lc_enable_dsl=true \
  --lc_enable_gui=true \
  --lc_enable_tests=true \
  --lc_enable_unity_build=false \
  -c
xmake

# Minimal build (no tests, no GUI)
xmake f -m release \
  --lc_enable_tests=false \
  --lc_enable_gui=false \
  --lc_enable_dsl=false \
  -c
xmake

# Debug build with tests
xmake f -m debug --lc_enable_tests=true -c
xmake
```

## Other Commands

```bash
# Clean build files
xmake clean

# Rebuild
xmake -r

# Run specific target
xmake run <target>

# List targets
xmake -l

# Install to directory
xmake install -o <dir>
```

## Common Mistakes to Avoid

### Invalid Flags
- `-v`, `-D`, `--diagnosis` are **NOT** valid xmake flags. The verbose flag is `--verbose` or `-v` (in newer versions), but use with caution.
- The project uses xmake 3.0.6+ with standard xmake syntax.

### Correct Build Commands
```bash
# Wrong (invalid flags)
xmake f -v -c -m debug

# Correct
xmake f -c -m debug
xmake build

# Verbose output (if needed)
xmake build --verbose
```

### Configuration Options
- Boolean options use `=true` or `=false` format: `--lc_enable_xir=true`
- Use `-c` flag to clean configuration cache before reconfiguring
- Use `-m` for build mode: `debug`, `release`, or `releasedbg`


## Build System Architecture

### Project Structure

XMake uses `xmake.lua` files in `src/` and most subdirectories. The root `xmake.lua` defines global options and includes subdirectory build files.

### Target Hierarchy

```
luisa-compute-include (header-only)
    ↓
luisa-compute-ext (third-party deps)
    ↓
luisa-compute-core
    ↓
luisa-compute-ast → luisa-compute-xir
    ↓
luisa-compute-runtime
    ↓
luisa-compute-dsl, luisa-compute-gui, luisa-compute-ir
    ↓
backends (plugins)
```

### Backend Plugin Pattern

Backends are built as `shared` targets with output names `luisa-backend-<name>`:
```lua
target("luisa-compute-backend-cuda")
    set_kind("shared")
    set_basename("luisa-backend-cuda")
    add_deps("luisa-compute-ast", "luisa-compute-runtime")
    -- ...
```

### Option System

Options are defined in root `xmake.lua` and consumed by subdirectory `xmake.lua` files:
```lua
option("lc_cuda_backend")
    set_default(true)
    set_showmenu(true)
    set_description("Enable CUDA backend")
```

### Rust Integration

XMake invokes `cargo build` via `os.run()` or `os.vrun()` in build rules:
```lua
rule("rust.build")
    on_build(function(target)
        os.run("cargo build --release")
    end)
```

### Key Differences from CMake

| Aspect | CMake | XMake |
|--------|-------|-------|
| Build file | `CMakeLists.txt` | `xmake.lua` |
| Backend output | `MODULE` library | `shared` target |
| Install paths | `install(TARGETS ...)` | `on_install()` rules |
| Rust integration | Custom commands + imported targets | Direct `os.run()` invocation |
| Configuration | `cmake -D` options | `xmake f --option=value` |
| Unity build | `LUISA_COMPUTE_ENABLE_UNITY_BUILD` | `lc_enable_unity_build` |
