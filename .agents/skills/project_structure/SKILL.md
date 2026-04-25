---
name: project_structure
description: Comprehensive project structure and architecture reference for the LuisaCompute codebase
---

# LuisaCompute Project Structure & Architecture

## Executive Summary

LuisaCompute is a high-performance, cross-platform GPU computing framework. The codebase follows a layered architecture: **Core** → **AST/IR** → **DSL/Runtime** → **Backends**. Source lives in `src/`; public headers in `include/luisa/`. The project uses dual build systems (CMake + XMake) and supports multiple language frontends (C++, Python, Rust).

---

## Top-Level Directory Map

```
src/
├── api/           # C API & runtime API layer
├── ast/           # Abstract Syntax Tree (DSL expression/statement/type system)
├── backends/      # Backend plugins (CUDA, DX, Metal, CPU, Vulkan, HIP, etc.)
├── clangcxx/      # C++ shader compiler frontend (Clang-based)
├── core/          # Foundation: types, math, logging, platform, STL wrappers
├── dsl/           # Embedded C++ DSL (kernel/callable authoring)
├── ext/           # Third-party dependencies (git submodules)
├── gui/           # Windowing, ImGui, framerate
├── ir/            # IR translation layer (AST↔IR, transforms)
├── osl/           # Open Shading Language parser support
├── py/            # Python bindings (pybind11 + pure Python luisa package)
├── runtime/       # Unified runtime: device, buffer, image, stream, RTX, raster
├── rust/          # Rust workspace: IR, backend impl, CPU backend, remote
├── tensor/        # Tensor operations & compute graph (fallback kernels)
├── tests/         # Test suites: unit tests, integration tests, examples
├── vstl/          # Virtual STL (custom containers, allocators, hashes)
└── xir/           # Extended IR (SSA, basic blocks, passes, translators)

include/luisa/     # Public headers mirroring src/ module layout
```

---

## Module Deep Dive

### `src/core/` — Foundation Layer

**Purpose**: Platform abstractions, math, logging, binary I/O, dynamic modules.

| File/Subdir | Role |
|-------------|------|
| `basic_types.cpp` | Vector/matrix type instantiations (`float2/3/4`, `float4x4`, etc.) |
| `basic_traits.cpp` | Compile-time type predicates for vectors/matrices |
| `logging.cpp` | spdlog-based logging infrastructure |
| `platform.cpp` | OS abstraction: paths, threads, DLL loading |
| `dynamic_module.cpp` | Cross-platform shared library loader |
| `binary_io.cpp`, `binary_file_stream.cpp` | Binary serialization helpers |
| `stl/` | Custom STL wrappers: `vector`, `string`, `unordered_map`, `optional`, `variant`, etc. |
| `generate_swizzles.py` | Generates vector swizzle code |

**Public headers**: `include/luisa/core/*`

---

### `src/vstl/` — Virtual STL

**Purpose**: High-performance custom containers and utilities not covered by `core/stl`.

| File | Role |
|------|------|
| `stack_allocator.cpp` | Arena/stack allocator |
| `string_builder.cpp` | Efficient string concatenation |
| `lmdb.cpp` | LMDB wrapper |
| `md5.cpp` | MD5 hashing |
| `v_guid.cpp` | GUID generation |

**Public headers**: `include/luisa/vstl/*` (hash maps, arenas, lockfree queues, ranges)

---

### `src/ast/` — Abstract Syntax Tree

**Purpose**: Represents DSL kernels/callables as a tree. The DSL traces C++ lambdas into AST nodes.

| File | Role |
|------|------|
| `expression.cpp` | AST expression nodes (literal, binary, unary, call, swizzle, member) |
| `statement.cpp` | AST statements (if, loop, switch, break, return, ray_query) |
| `type.cpp` | Type system (scalars, vectors, matrices, buffers, textures, structs) |
| `function.cpp` | Function representation (kernel/callable metadata) |
| `function_builder.cpp` | Builder API for manual AST construction |
| `variable.cpp` | Variable and reference nodes |
| `op.cpp` | Operator enums (`BinaryOp`, `UnaryOp`, `CallOp`) |
| `external_function.cpp` | External function references |
| `callable_library.cpp` | Callable library serialization |
| `constant_data.cpp` | Constant data embedding |
| `ast2json.cpp` | AST → JSON serialization |

**Public headers**: `include/luisa/ast/*` (`function_builder.h`, `type.h`, `expression.h`, `statement.h`, `op.h`)

---

### `src/xir/` — Extended IR (Next-Gen Compiler IR)

**Purpose**: SSA-based IR with basic blocks, instructions, and optimization passes. Receives AST via `ast2xir` translator.

| Subdir | Role |
|--------|------|
| `instructions/` | 30+ instruction types: arithmetic, memory (load/store/alloca), control flow (if/loop/switch/branch/phi), resource (buffer/texture/ray_query), autodiff, atomic, debug |
| `metadata/` | Debug metadata: names, locations, comments, curve basis |
| `passes/` | Optimization/analysis passes: DCE, mem2reg, SROA, autodiff, outline, dom-tree, GEP tracing/transpose, local store forward/load elimination, ray-query lowering, unused callable removal |
| `translators/` | `ast2xir`, `xir2json`, `json2xir`, `xir2text` |
| `tests/` | Unit tests for XIR passes |

**Key classes**: `Module`, `Function`, `BasicBlock`, `Instruction`, `Value`, `Use`, `Builder`

**Public headers**: `include/luisa/xir/*`

---

### `src/ir/` — IR Bridge Layer

**Purpose**: Transforms between AST and an older JSON-serializable IR; also hosts high-level transforms.

| File | Role |
|------|------|
| `ast2ir.cpp` | AST → IR translation |
| `ir2ast.cpp` | IR → AST round-trip |
| `transform.cpp` | IR-level transforms |

**Public headers**: `include/luisa/ir/*`

---

### `src/dsl/` — Embedded Domain-Specific Language

**Purpose**: Allows writing GPU kernels in C++ via lambda tracing.

| File | Role |
|------|------|
| `func.cpp` | `Kernel1D/2D/3D`, `Callable` definitions |
| `builtin.cpp` | Built-in function dispatch (`dispatch_id`, `thread_id`, math) |
| `local.cpp` | Local variable and reference handling |
| `resource.cpp` | Buffer, image, volume, bindless array DSL wrappers |
| `sugar.cpp` | Sugar macro implementations (`$if`, `$for`, `$while`) |
| `polymorphic.cpp` | Polymorphic dispatch support |
| `soa.cpp` | Structure-of-arrays helpers |
| `dispatch_indirect.cpp` | Indirect dispatch buffer DSL bindings |
| `rtx/` | Ray tracing DSL: `Accel`, `Ray`, `RayQuery`, `Curve`, `TriangleHit` |
| `raster/` | Rasterization DSL: `RasterKernel` |

**Public headers**: `include/luisa/dsl/*` (`syntax.h`, `sugar.h`, `func.h`, `struct.h`, `var.h`)

---

### `src/runtime/` — Unified Runtime

**Purpose**: Resource management, command scheduling, device abstraction (RHI pattern).

| File/Subdir | Role |
|-------------|------|
| `device.cpp` | Device creation, backend plugin loading |
| `context.cpp` | Context initialization, backend enumeration |
| `stream.cpp` | Command stream submission |
| `command_list.cpp` | Command batching |
| `buffer.cpp`, `image.cpp`, `volume.cpp` | GPU memory resources |
| `byte_buffer.cpp`, `sparse_buffer.cpp`, `sparse_texture.cpp` | Untyped/sparse memory |
| `bindless_array.cpp` | Bindless resource array management |
| `swapchain.cpp` | Window presentation |
| `event.cpp` | Binary/timeline synchronization events |
| `builtin_kernel.cpp` | Built-in kernel utilities |
| `mipmap.cpp` | Mipmap generation commands |
| `rhi/` | **RHI interfaces**: `device_interface.h`, `command.h`, `command_encoder.h`, `resource.h`, `pixel.h` |
| `rtx/` | Ray tracing runtime: `accel.cpp`, `mesh.cpp`, `curve.cpp`, `motion_instance.cpp`, `procedural_primitive.cpp` |
| `raster/` | Rasterization runtime: `raster.cpp`, `depth_buffer.cpp` |
| `remote/` | Remote device client/server interfaces |

**Public headers**: `include/luisa/runtime/*`

---

### `src/backends/` — Backend Plugins

**Architecture**: Each backend is a dynamically loaded plugin (`luisa-backend-<name>.dll/.so`). Loaded by `Context` at runtime.

| Backend | Path | Technology |
|---------|------|------------|
| **CUDA** | `src/backends/cuda/` | NVRTC + OptiX ray tracing + CUDA driver API |
| **DirectX** | `src/backends/dx/` | DirectX 12 + DXR + HLSL DXC compiler |
| **Metal** | `src/backends/metal/` | Metal 3 + MSL compiler |
| **CPU** | `src/backends/cpu/` | Rust-based CPU backend (via `src/rust/`) |
| **Vulkan** | `src/backends/vk/` | Vulkan compute/graphics + SPIR-V |
| **HIP** | `src/backends/hip/` | AMD HIP runtime |
| **Remote** | `src/backends/remote/` | Network-distributed compute |
| **Fallback** | `src/backends/fallback/` | Reference/fallback interpreter |
| **Toy C** | `src/backends/toy_c/` | Minimal C code generation backend |
| **Validation** | `src/backends/validation/` | Validation/debugging layer |
| **Common** | `src/backends/common/` | Shared code: Vulkan swapchain, OIDN denoiser, LLVM linking helpers, HLSL builtins |

**Backend internal pattern**: Each backend implements `DeviceInterface` (from `runtime/rhi/`) and provides:
- Codegen: AST/XIR → backend shader source (CUDA C++, HLSL, MSL, SPIR-V, etc.)
- Compiler: Native shader compilation (NVRTC, DXC, etc.)
- Resources: Backend-specific buffer/image/accel implementations
- Commands: Command encoder translating RHI commands to API calls

---

### `src/rust/` — Rust Workspace

**Purpose**: IR implementation, CPU backend codegen, and language bindings.

| Crate | Role |
|-------|------|
| `luisa_compute_ir` | Core IR: AST→IR, IR analysis (use/def), transforms (DCE, inliner, SSA, autodiff, vectorize), serialization |
| `luisa_compute_ir_v2` | IR v2 bindings and conversion |
| `luisa_compute_ir_staticlib` | Static library wrapper for C++ linking |
| `luisa_compute_api_types` | C API type definitions shared with C++ |
| `luisa_compute_backend` | Backend proxy/message protocol |
| `luisa_compute_backend_impl` | **CPU backend**: LLVM JIT, C++ codegen, texture sampling, acceleration structures, remote backend |
| `luisa_compute_cpu_kernel_defs` | CPU kernel ABI definitions |

**Integration**: Rust crates are built via CMake/Cargo interop and linked into C++ targets.

---

### `src/api/` — C & Runtime API

**Purpose**: Stable C API for external language bindings and runtime services.

| File | Role |
|------|------|
| `runtime.cpp` | C runtime API implementation |
| `logging.cpp` | C logging API |
| `gen_rust_binding.py` | Generates Rust bindings from API headers |
| `gen_remote_rpc_types.py` | Generates remote RPC type stubs |

**Public headers**: `include/luisa/api/*`

---

### `src/py/` — Python Frontend

**Purpose**: Python bindings and pure-Python wrapper library.

| File | Role |
|------|------|
| `lcapi.cpp` | Main pybind11 module entry point |
| `export_*.cpp` | Per-component exports: runtime, DSL, GUI, math types, ops, buffers, vectors |
| `managed_*.cpp/h` | Managed Python wrappers for accel, bindless, device, collector |
| `image_util.cpp` | Image I/O utilities |
| `interop.cpp/h` | PyTorch/DLPack interop |
| `py_stream.cpp/h` | Python stream wrapper |
| `luisa/` | **Pure Python package**: `__init__.py`, `buffer.py`, `accel.py`, `autodiff.py`, `gui.py`, `types.py`, `vector.py`, etc. |

---

### `src/tensor/` — Tensor & Compute Graph

**Purpose**: High-level tensor operations and automatic kernel generation.

| File/Subdir | Role |
|-------------|------|
| `tensor.cpp` | Tensor class implementation |
| `expression.cpp` | Tensor expression DAG |
| `graph.cpp` | Compute graph builder |
| `pass/` | Graph passes: expression topo-sort, shader manager |
| `fallback/` | Fallback CPU implementations: matmul, softmax, set_value |

**Public headers**: `include/luisa/tensor/*`

---

### `src/clangcxx/` — C++ Shader Compiler

**Purpose**: Compiles C++ code (not DSL) directly to GPU shaders using Clang/libTooling.

| File/Subdir | Role |
|-------------|------|
| `src/` | Compiler frontend implementation |
| `llvm/` | LLVM integration helpers |

**Public headers**: `include/luisa/clangcxx/*`

---

### `src/osl/` — Open Shading Language

**Purpose**: Parses OSL bytecode (`.oso`) for shader interoperability.

| File | Role |
|------|------|
| `oso_parser.cpp` | OSO format parser |
| `shader.cpp` | OSL shader wrapper |
| `type.cpp`, `symbol.cpp`, `instruction.cpp`, `literal.cpp`, `hint.cpp` | OSL IR representation |

**Public headers**: `include/luisa/osl/*`

---

### `src/gui/` — GUI & Presentation

**Purpose**: Cross-platform windowing and ImGui integration.

| File | Role |
|------|------|
| `window.cpp` | OS window abstraction |
| `imgui_window.cpp` | ImGui render loop integration |
| `framerate.cpp` | FPS counter |

**Public headers**: `include/luisa/gui/*`

---

### `src/ext/` — Third-Party Dependencies

All managed as git submodules.

| Dependency | Usage |
|------------|-------|
| `EASTL` | Electronic Arts STL (optional container backend) |
| `glfw` | Windowing for GUI/tests |
| `imgui` | Immediate mode GUI |
| `pybind11` | Python C++ bindings |
| `spdlog` | Logging backend |
| `reproc` | Process spawning |
| `stb` | STB image libraries |
| `volk` | Vulkan meta-loader |
| `yyjson` | High-performance JSON |
| `xxhash` | Fast hashing |
| `magic_enum` | Enum reflection |
| `marl` | Fiber/task library |
| `half` | Half-precision float |
| `HIPRT` | AMD HIP ray tracing |
| `liblmdb` | Lightning memory-mapped database |

---

### `src/tests/` — Test Suites

| Subdir | Content |
|--------|---------|
| `for_agent/` | Core library unit tests (traits, types, I/O, math, allocators) |
| `next/test/feat/` | Feature tests by layer: `ast/`, `dsl/`, `ir/`, `runtime/`, `tensor/` |
| `next/example/` | Example apps: gallery (fluid sim, procedural, render), usage demos |
| `python/` | Python frontend tests (path tracing, MPM, RTX, Game of Life) |
| `cxx_shaders/` | C++ shader standard library tests (experimental C++ frontend) |
| `clangcxx_compiler/` | ClangCXX compiler tests |
| `common/` | Shared test utilities: doctest, math helpers, OBJ/EXR loaders |
| `transient_resource_device/` | Transient resource allocation tests |
| *(root)* | Integration tests: `test_helloworld`, `test_path_tracing`, `test_rtx`, `test_raster`, `test_tensor`, `test_dsl`, `test_autodiff`, `test_fp8`, etc. |

---

## Build System Architecture

### CMake (Primary)

- **Root**: `src/CMakeLists.txt` adds subdirectories in dependency order.
- **Targets**: Each module produces `luisa-compute-<name>` (static/shared library).
- **Alias**: `luisa::compute` interface target links all modules.
- **Backends**: Built as `MODULE` (plugins) named `luisa-backend-<name>`.
- **Options**: `LUISA_COMPUTE_ENABLE_CUDA`, `_DX`, `_METAL`, `_CPU`, `_VULKAN`, `_HIP`, `_DSL`, `_RUST`, etc.
- **Install**: `luisa_compute_install(target)` helper for consistent packaging.

### XMake (Secondary)

- `xmake.lua` files in `src/` and most subdirectories.
- Used by some developers for faster incremental builds.

### Bootstrap Script

- `bootstrap.py` at repo root automates dependency installation and CMake configuration.

---

## Compiler Pipeline Flow

```
User C++ DSL / Python
         │
         ▼
┌─────────────────┐
│   DSL Tracing   │  src/dsl/ — captures C++ lambdas into AST
│  (src/dsl/)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│       AST       │  src/ast/ — expression/statement tree
│  (src/ast/)     │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│  XIR   │ │   IR     │  src/xir/ (new)  /  src/ir/ (legacy bridge)
│transl. │ │transl.   │
└───┬────┘ └────┬─────┘
    │           │
    ▼           ▼
┌─────────────────┐
│  Backend Codegen│  src/backends/<name>/ — AST/XIR → native shader source
│  + Compilation  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPU Execution  │  src/runtime/ — command submission via Stream
│  (src/runtime/) │
└─────────────────┘
```

**Rust IR path**: `src/rust/luisa_compute_ir/` performs additional IR transforms (autodiff, DCE, SSA, vectorize) before backend codegen.

---

## Public Header Organization (`include/luisa/`)

Headers mirror `src/` modules. Key umbrella headers:

| Header | Includes |
|--------|----------|
| `<luisa/luisa-compute.h>` | Core + AST + DSL + Runtime + GUI (sugar excluded) |
| `<luisa/dsl/syntax.h>` | DSL core |
| `<luisa/dsl/sugar.h>` | Sugar macros (`$if`, `$for`, etc.) |
| `<luisa/runtime/context.h>` | Runtime entry point |
| `<luisa/runtime/device.h>` | Device and resource creation |

---

## Key Design Patterns

1. **RHI (Rendering Hardware Interface)**: `src/runtime/rhi/` abstracts all GPU APIs into common interfaces.
2. **Plugin Architecture**: Backends are dynamic modules loaded at runtime by `Context`.
3. **RAII Resources**: All runtime objects (`Buffer`, `Image`, `Stream`, `Accel`) are move-only RAII handles.
4. **Command-Based Execution**: Work is encoded as `Command` objects batched into `CommandList`s and submitted to `Stream`s.
5. **DSL Tracing**: Operator overloading and lambda capture build an AST at kernel definition time.
6. **Dual IR**: AST is the frontend tree; XIR is the SSA backend IR with optimization passes.
7. **Rust + C++ Hybrid**: Performance-critical IR and CPU backend use Rust; C++ handles API and DSL.

---

## Naming Conventions

| Convention | Example | Meaning |
|------------|---------|---------|
| `luisa-compute-<module>` | `luisa-compute-core` | CMake target name |
| `luisa-backend-<name>` | `luisa-backend-cuda` | Backend plugin binary |
| `lc_*_pch.h` | `lc_core_pch.h` | Precompiled header for module |
| `test_<feature>.cpp` | `test_path_tracing.cpp` | Integration test |
| `export_<component>.cpp` | `export_runtime.cpp` | Python binding export |
| `xmake.lua` | — | XMake build file per directory |

---

## File Count Summary (approximate)

| Layer | Directories | Key Files |
|-------|-------------|-----------|
| Core + VSTL | 3 | ~30 source files |
| AST + IR + XIR | 5 | ~80 source files |
| DSL | 4 | ~15 source files |
| Runtime + RHI | 7 | ~40 source files |
| Backends | 11 | ~200+ source files |
| Rust | 7 crates | ~50 Rust source files |
| Python | 2 | ~25 C++ + ~30 Python files |
| Tests | 6+ | ~150 test files |
| Ext | 15+ | External submodules |

---

## Maintenance Notes

- Adding a new backend: Create `src/backends/<name>/`, implement `DeviceInterface`, register in `src/backends/CMakeLists.txt`.
- Adding a new XIR pass: Add to `src/xir/passes/`, register in `src/xir/CMakeLists.txt`.
- Adding a new runtime resource: Define in `src/runtime/rhi/resource.h`, implement in each backend, expose in `src/runtime/` and `include/luisa/runtime/`.
- The `tensor/` module is currently opt-in and less mature than DSL/Runtime.
- `clangcxx/` is experimental for full C++ shader compilation (not DSL tracing).
