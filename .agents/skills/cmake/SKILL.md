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
