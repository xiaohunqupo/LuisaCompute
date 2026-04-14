---
name: cmake
---

# CMake Build Guide

## Requirements

- CMake 3.26+
- Ninja (recommended)
- C++20 compiler (MSVC/Clang/GCC)

## Quick Start

```bash
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix dist
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | - | `Release` or `Debug` |
| `LUISA_COMPUTE_ENABLE_DSL` | ON | C++ DSL |
| `LUISA_COMPUTE_ENABLE_CUDA` | ON | CUDA backend |
| `LUISA_COMPUTE_ENABLE_METAL` | ON | Metal (macOS) |
| `LUISA_COMPUTE_ENABLE_DX` | ON | DirectX (Windows) |
| `LUISA_COMPUTE_ENABLE_VULKAN` | ON | Vulkan backend |
| `LUISA_COMPUTE_ENABLE_CPU` | ON | CPU backend |
| `LUISA_COMPUTE_ENABLE_REMOTE` | ON | Remote backend |
| `LUISA_COMPUTE_ENABLE_FALLBACK` | ON | Fallback backend |
| `LUISA_COMPUTE_ENABLE_GUI` | ON | GUI support |
| `LUISA_COMPUTE_ENABLE_UNITY_BUILD` | OFF | Unity build |
| `LUISA_COMPUTE_ENABLE_SANITIZERS` | OFF | Address/UB sanitizers |
| `LUISA_COMPUTE_USE_SYSTEM_LIBS` | OFF | Prefer system libs |

## Platform-Specific

### Linux
```bash
export CC=clang-20 CXX=clang++-20
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release
```

### macOS
```bash
export PATH="$PATH:/opt/homebrew/opt/llvm/bin"
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export SDKROOT=$(xcrun --show-sdk-path)
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release
```

### Windows

**Recommended (Python bootstrap):**
```python
import bootstrap
bootstrap.prepare_msvc_environment()
```

**Manual (VS Developer Command Prompt required):**
```cmd
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release
```

## CI Example

```bash
cmake -S . -B build -G Ninja \
  -D LUISA_COMPUTE_ENABLE_RUST=OFF \
  -D LUISA_COMPUTE_ENABLE_REMOTE=OFF \
  -D LUISA_COMPUTE_ENABLE_CPU=OFF \
  -D CMAKE_BUILD_TYPE=Release
cmake --build build
```
