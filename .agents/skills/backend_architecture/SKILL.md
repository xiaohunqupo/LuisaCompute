---
name: backend_architecture
description: Backend plugin architecture, DeviceInterface API, dynamic loading, command encoding, and backend registration patterns in LuisaCompute
---

# LuisaCompute Backend Plugin Architecture

## Overview

Backends are dynamically loaded plugins that implement the `DeviceInterface` API. Each backend is a separate shared library (`luisa-backend-<name>.dll/.so/.dylib`) discovered and loaded at runtime by `Context`.

---

## DeviceInterface API Surface

**Header**: `include/luisa/runtime/rhi/device_interface.h`

All backends inherit from `DeviceInterface` and must implement:

### Resource Lifecycle (Handle-based)

| Resource | Create | Destroy |
|----------|--------|---------|
| Buffer | `create_buffer(Type*, size, external)` | `destroy_buffer(uint64_t)` |
| Texture | `create_texture(format, dim, w, h, d, mips, ...)` | `destroy_texture(uint64_t)` |
| Bindless Array | `create_bindless_array(size, type)` | `destroy_bindless_array(uint64_t)` |
| Stream | `create_stream(StreamTag)` | `destroy_stream(uint64_t)` |
| Event | `create_event()` | `destroy_event(uint64_t)` |
| Shader | `create_shader(ShaderOption, Function/IR)` | `destroy_shader(uint64_t)` |
| Mesh | `create_mesh(AccelOption)` | `destroy_mesh(uint64_t)` |
| Curve | `create_curve(AccelOption)` | `destroy_curve(uint64_t)` |
| Procedural Primitive | `create_procedural_primitive(AccelOption)` | `destroy_procedural_primitive(uint64_t)` |
| Motion Instance | `create_motion_instance(AccelMotionOption)` | `destroy_motion_instance(uint64_t)` |
| Accel | `create_accel(AccelOption)` | `destroy_accel(uint64_t)` |
| Swapchain | `create_swapchain(SwapchainOption, stream)` | `destroy_swapchain(uint64_t)` |

### Execution

| Method | Purpose |
|--------|---------|
| `dispatch(stream_handle, CommandList)` | Submit commands for execution |
| `synchronize_stream(stream_handle)` | Host-wait for stream |
| `set_stream_log_callback(stream_handle, cb)` | Set per-stream logging |

### Queries & Extensions

| Method | Purpose |
|--------|---------|
| `native_handle()` | Underlying API handle (CUcontext, VkDevice, etc.) |
| `compute_warp_size()` | Warp/wavefront size (32 for CUDA, 1 for CPU/fallback) |
| `memory_granularity()` | Memory allocation alignment |
| `query(property)` | Device property queries |
| `extension(name)` | Retrieve device extension interface |
| `set_name(handle, name)` | Debug naming |

### Event Synchronization

| Method | Purpose |
|--------|---------|
| `signal_event(handle, stream, value)` | Signal timeline event |
| `wait_event(handle, stream, value)` | Wait on timeline event |
| `is_event_completed(handle, value)` | Poll event completion |
| `synchronize_event(handle, value)` | Host-wait on event |

---

## Dynamic Loading Mechanism

**File**: `src/runtime/context.cpp`

### Backend Discovery

At startup, `Context` scans the runtime directory for files matching:
- `luisa-backend-*.so` (Linux)
- `luisa-backend-*.dll` (Windows)
- `luisa-backend-*.dylib` (macOS)
- `libluisa-backend-*` (MinGW)

### Backend Loading Flow

```cpp
const BackendModule &load_backend(const luisa::string &backend_name) {
    // 1. Check installed_backends list
    // 2. Load dynamic library: luisa-backend-<name>.<ext>
    // 3. Validate version: backend_version() must match LUISA_COMPUTE_VERSION
    // 4. Extract function pointers:
    //    - creator: Device::Creator ("create")
    //    - deleter: Device::Deleter ("destroy")
    //    - backend_device_names: BackendDeviceNames ("backend_device_names")
}
```

### Device Creation

```cpp
Device Context::create_device(backend_name, settings, enable_validation) {
    auto &m = _impl->load_backend(backend_name);
    auto interface = m.creator(Context{_impl}, settings);
    if (enable_validation) {
        interface = wrap_with_validation(interface);
    }
    return Device{handle};
}
```

---

## Backend Registration Pattern

Every backend must export **three C functions** with `LUISA_EXPORT_API`:

```cpp
// File: src/backends/<name>/<name>_device.cpp

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(
    luisa::compute::Context &&ctx,
    const luisa::compute::DeviceConfig *) noexcept {
    return luisa::new_with_allocator<MyBackendDevice>(std::move(ctx));
}

LUISA_EXPORT_API void destroy(
    luisa::compute::DeviceInterface *device) noexcept {
    luisa::delete_with_allocator(device);
}

LUISA_EXPORT_API void backend_device_names(
    luisa::vector<luisa::string> &names) noexcept {
    names.clear();
    names.emplace_back("cuda"); // or "cpu", "dx", "vk", etc.
}
```

Plus version export via `src/backends/common/export_version.inl.h`:
```cpp
LUISA_EXPORT_API int backend_version() { 
    return LUISA_COMPUTE_VERSION; 
}
```

---

## Command Encoder Pattern

Commands are dispatched via the **Visitor Pattern** (`MutableCommandVisitor`):

```cpp
class MyCommandEncoder : public MutableCommandVisitor {
    MyStream *_stream;
    
    void visit(BufferUploadCommand *cmd) noexcept override;
    void visit(BufferDownloadCommand *cmd) noexcept override;
    void visit(BufferCopyCommand *cmd) noexcept override;
    void visit(ShaderDispatchCommand *cmd) noexcept override;
    void visit(AccelBuildCommand *cmd) noexcept override;
    void visit(MeshBuildCommand *cmd) noexcept override;
    void visit(BindlessArrayUpdateCommand *cmd) noexcept override;
    void visit(CustomCommand *cmd) noexcept override;
    
    void commit(CommandList::CallbackContainer &&user_callbacks) noexcept;
};
```

Stream architecture:
- Each backend has a `*Stream` class owning the native stream/queue
- `stream->dispatch(CommandList)` visits all commands via the encoder
- Callbacks are collected and executed after GPU work completes

---

## Resource Handle Pattern

Resources are created as backend-specific classes and returned as opaque `uint64_t` handles:

```cpp
auto buffer = new_with_allocator<CUDABuffer>(size);
return {
    .handle = reinterpret_cast<uint64_t>(buffer),
    .native_handle = reinterpret_cast<void *>(buffer->device_address())
};
```

Destruction recovers the typed pointer:
```cpp
void destroy_buffer(uint64_t handle) noexcept {
    auto buffer = reinterpret_cast<CUDABuffer *>(handle);
    delete_with_allocator(buffer);
}
```

---

## Files Every Backend Needs

### Minimal Backend

| File | Purpose |
|------|---------|
| `<backend>_device.h` | Device class inheriting `DeviceInterface` |
| `<backend>_device.cpp` | Device implementation + exported C functions |
| `<backend>_stream.h/cpp` | Stream/command queue implementation |
| `<backend>_buffer.h/cpp` | Buffer resource wrapper |
| `<backend>_texture.h/cpp` | Texture/image resource wrapper |
| `<backend>_shader.h/cpp` | Shader/kernel compilation and management |
| `<backend>_event.h/cpp` | Synchronization primitives |
| `CMakeLists.txt` | Build configuration |

### Common Helper Files

| File | Purpose |
|------|---------|
| `<backend>_command_encoder.h/cpp` | Command visitor implementation |
| `<backend>_accel.h/cpp` | Ray tracing acceleration structures |
| `<backend>_mesh.h/cpp` | Mesh geometry |
| `<backend>_bindless_array.h/cpp` | Bindless resource arrays |
| `<backend>_swapchain.h/cpp` | Display presentation |

### Shared Common Files

From `src/backends/common/`:
- `default_binary_io.h/cpp` — Shader caching I/O
- `export_version.inl.h` — Version export include
- `hlsl/builtin/` — HLSL builtin headers/bytecode (DX backend)
- `vulkan_swapchain.h/cpp` — Shared Vulkan swapchain

---

## Backend CMake Patterns

### Registration Function

From `src/backends/CMakeLists.txt`:

```cmake
function(luisa_compute_add_backend name)
    cmake_parse_arguments(BACKEND "" "SUPPORT_DIR;BUILTIN_DIR" "SOURCES" ${ARGN})
    
    add_library(luisa-compute-backend-${name} MODULE ${BACKEND_SOURCES})
    
    target_link_libraries(luisa-compute-backend-${name} PRIVATE
        luisa-compute-ast
        luisa-compute-runtime
        luisa-compute-gui)
    
    set_target_properties(luisa-compute-backend-${name} PROPERTIES
        OUTPUT_NAME luisa-backend-${name})
    
    install(TARGETS luisa-compute-backend-${name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()
```

### Minimal Example (CPU backend)

```cmake
set(LUISA_COMPUTE_CPU_SOURCES
    ../common/rust_device_common.cpp ../common/rust_device_common.h
    cpu_device.h cpu_device.cpp)
    
luisa_compute_add_backend(cpu SOURCES ${LUISA_COMPUTE_CPU_SOURCES})

target_link_libraries(luisa-compute-backend-cpu PRIVATE
    luisa-compute-vulkan-swapchain
    luisa-compute-rust-meta
    luisa_compute_backend_impl)
```

### Complex Example (Fallback backend)

```cmake
find_package(LLVM CONFIG)
find_package(embree CONFIG)

if (LLVM_FOUND AND embree_FOUND)
    set(LUISA_FALLBACK_BACKEND_SOURCES
        fallback_device.cpp fallback_device.h
        fallback_stream.cpp fallback_stream.h
        # ... 20+ more files
    )
    
    luisa_compute_add_backend(fallback SOURCES ${LUISA_FALLBACK_BACKEND_SOURCES})
    
    target_link_libraries(luisa-compute-backend-fallback PRIVATE
        luisa-compute-xir
        luisa-compute-vulkan-swapchain
        embree)
    
    luisa_compute_link_llvm_into_backend(fallback REQUIRED
        COMPONENTS core executionengine ... )
endif()
```

### Advanced Patterns (CUDA backend)

- Custom embedding commands for builtin kernels (`luisa_embed_device_lib`)
- Device runtime library embedding
- Optional components: nvCOMP, NVTT, OIDN
- Standalone compiler executable
- Platform-specific linking

---

## Backend Codegen Pipeline

Within each backend, the compilation flow is:

```
AST/XIR Function
       │
       ▼
┌─────────────────┐
│  Backend Codegen │  AST/XIR → native shader source (CUDA C++, HLSL, MSL, SPIR-V)
│  (per-backend)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Native Compiler  │  NVRTC, DXC, Metal compiler, SPIR-V tools, etc.
│ (per-backend)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Shader Object   │  PTX, DXIL, metallib, SPIR-V binary
│  (cached via     │  Cached via BinaryIO interface
│   BinaryIO)      │
└─────────────────┘
```

---

## Key Design Decisions

1. **Plugin Architecture**: Backends are shared libraries loaded at runtime, not linked directly. This allows distributing backends separately and loading only what's available.
2. **Handle-based Resources**: All resources are opaque `uint64_t` handles for ABI stability across backend boundaries.
3. **Visitor Pattern**: Commands use double dispatch via `MutableCommandVisitor` for type-safe command handling.
4. **Version Checking**: Strict version matching between runtime and backend prevents ABI mismatches.
5. **Validation Layer**: Optional wrap-around validation can be enabled at device creation.
6. **BinaryIO Interface**: Pluggable shader caching system via `default_binary_io`.
