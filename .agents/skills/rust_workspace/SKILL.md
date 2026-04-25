---
name: rust_workspace
description: Rust workspace architecture, IR data structures, compiler transforms, CPU backend, FFI integration with C++, and crate dependencies in LuisaCompute
---

# LuisaCompute Rust Workspace

## Overview

The Rust workspace (`src/rust/`) implements the legacy IR, compiler optimization passes, and the CPU/Remote backends. It integrates with C++ via FFI using `cbindgen` for header generation and static libraries for linking.

---

## Workspace Structure

```
src/rust/
├── Cargo.toml (workspace root)
├── luisa_compute_ir/           # Core IR data structures and transforms
├── luisa_compute_ir_staticlib/ # Static library wrapper for C++ linking
├── luisa_compute_ir_v2/        # IR v2 bindings
├── luisa_compute_api_types/    # Shared C++/Rust API types
├── luisa_compute_cpu_kernel_defs/ # CPU kernel runtime definitions
├── luisa_compute_backend/      # Backend trait definitions
└── luisa_compute_backend_impl/ # Backend implementations (CPU, Remote)
```

---

## Crate Dependency Graph

```
luisa_compute_api_types (base)
    ↑
    ├── luisa_compute_ir
    │       ├── luisa_compute_ir_staticlib
    │       ├── luisa_compute_ir_v2
    │       └── luisa_compute_backend
    │               └── luisa_compute_backend_impl
    │                       └── (uses luisa_compute_cpu_kernel_defs)
    │
    └── luisa_compute_cpu_kernel_defs
```

---

## Crate Responsibilities

### `luisa_compute_api_types`

**Purpose**: Defines shared types between C++ and Rust via FFI.

- **Crate Type**: `staticlib` + `rlib`
- **Key Types**:
  - Resource handles: `Buffer`, `Texture`, `Stream`, `Device`, `Shader`, `Accel`, `Mesh`
  - Command types: `BufferUploadCommand`, `ShaderDispatchCommand`, `AccelBuildCommand`
  - Pixel formats: `PixelStorage`, `PixelFormat` (28 variants including BC compression)
  - Sampler types: `SamplerFilter`, `SamplerAddress`
  - Ray tracing: `AccelOption`, `AccelBuildModification`, `CurveBasis`
  - Device interface: `DeviceInterface` (vtable of function pointers)
  - Denoiser extension types
- **Build Script**: Uses `cbindgen` to generate `api_types.hpp` and `api_types.h` for C++

### `luisa_compute_ir`

**Purpose**: Core IR data structures and compiler transforms.

- **Crate Type**: `rlib`
- **Dependencies**: `luisa_compute_api_types`, `half`, `serde`, `bincode`, `indexmap`

**Core Data Structures** (`src/ir.rs`):
```rust
pub enum Type {
    Void, Primitive(Primitive), Vector(VectorType), Matrix(MatrixType),
    Struct(StructType), Array(ArrayType), Opaque(CBoxedSlice<u8>)
}

pub struct Node {
    pub type_: CArc<Type>,
    pub next: NodeRef,
    pub prev: NodeRef,
    pub instruction: CArc<Instruction>,
}

pub enum Instruction {
    Local { init: NodeRef },
    Update { var: NodeRef, value: NodeRef },
    Call(Func, CBoxedSlice<NodeRef>),
    Phi(CBoxedSlice<PhiIncoming>),
    If { cond: NodeRef, true_branch: Pooled<BasicBlock>, false_branch: Pooled<BasicBlock> },
    Loop { body: Pooled<BasicBlock>, cond: NodeRef },
    GenericLoop { prepare, cond, body, update: Pooled<BasicBlock> },
    Switch { value: NodeRef, default: Pooled<BasicBlock>, cases: CBoxedSlice<SwitchCase> },
    AdScope { body: Pooled<BasicBlock>, forward: bool, n_forward_grads: usize },
    RayQuery { ray_query: NodeRef, on_triangle_hit: Pooled<BasicBlock>, on_procedural_hit: Pooled<BasicBlock> },
    // ... 20+ more variants
}

pub enum Func {
    // 180+ variants including arithmetic, math, memory, atomic, warp, ray tracing, AD
}
```

**Memory Management**:
- `CArc<T>`: C-compatible atomic reference counting
- `CBoxedSlice<T>`: C-compatible boxed slice
- `Pool<T>`: Chunked memory pool for nodes
- `ModulePools`: Separate pools for nodes and basic blocks

### `luisa_compute_ir_staticlib`

**Purpose**: Static library wrapper for C++ linking.

- **Crate Type**: `staticlib`
- **Exports**: All `luisa_compute_ir` symbols as static library

### `luisa_compute_ir_v2`

**Purpose**: IR v2 dynamic loading support.

- **Dependencies**: `libloading`, `luisa_compute_ir`

### `luisa_compute_cpu_kernel_defs`

**Purpose**: Runtime type definitions for CPU kernels.

- **Key Types**:
  - `KernelFnArgs`: Arguments passed to CPU kernel functions
  - `BufferView`, `Texture`, `Accel`: Runtime resource wrappers
  - `Ray`, `Hit`, `RayQuery`: Ray tracing structures
  - `CpuCustomOp`: Custom CPU operation callback

### `luisa_compute_backend`

**Purpose**: Backend trait abstraction.

- **Key Types**:
  - `Backend` trait: Core interface with 20+ methods for resource management
  - `Context`: Loads backend DLLs (`luisa-api.dll`/`libluisa-api.so`)
  - `ProxyBackend`: Dynamic dispatch wrapper

### `luisa_compute_backend_impl`

**Purpose**: Concrete backend implementations.

- **Crate Type**: `cdylib`
- **Features**: `cpu`, `remote`

**CPU Backend** (`cpu/`):
- `RustBackend`: Main CPU backend using Rayon thread pool
- `shader.rs`: JIT compilation via Clang/LLVM
- `codegen/cpp.rs`: C++ code generation from IR
- `accel.rs`: Embree ray tracing integration
- `stream.rs`: Command queue management
- `texture.rs`, `resource.rs`: Resource management

---

## Rust IR Data Structures

### Type System (`src/ir.rs`)

```rust
pub enum Type {
    Void,
    Primitive(Primitive),
    Vector(VectorType),
    Matrix(MatrixType),
    Struct(StructType),
    Array(ArrayType),
    Opaque(CBoxedSlice<u8>),
}

pub enum Primitive {
    Bool,
    Int8, Int16, Int32, Int64,
    Uint8, Uint16, Uint32, Uint64,
    Float16, Float32, Float64,
}
```

### Node Structure

Intrusive doubly-linked list for cache-friendly traversal:
```rust
pub struct Node {
    pub type_: CArc<Type>,
    pub next: NodeRef,
    pub prev: NodeRef,
    pub instruction: CArc<Instruction>,
}
```

### Control Flow

```rust
pub enum Instruction {
    Local { init: NodeRef },
    Update { var: NodeRef, value: NodeRef },
    Call(Func, CBoxedSlice<NodeRef>),
    Phi(CBoxedSlice<PhiIncoming>),
    If { cond: NodeRef, true_branch: Pooled<BasicBlock>, false_branch: Pooled<BasicBlock> },
    Loop { body: Pooled<BasicBlock>, cond: NodeRef },
    GenericLoop { prepare, cond, body, update: Pooled<BasicBlock> },
    Switch { value: NodeRef, default: Pooled<BasicBlock>, cases: CBoxedSlice<SwitchCase> },
    AdScope { body: Pooled<BasicBlock>, forward: bool, n_forward_grads: usize },
    RayQuery { ray_query: NodeRef, on_triangle_hit: Pooled<BasicBlock>, on_procedural_hit: Pooled<BasicBlock> },
    // ...
}
```

### Function Registry (`Func` enum)

180+ built-in functions:
- **Math**: `Add`, `Mul`, `Sin`, `Cos`, `Exp`, `Log`, `Sqrt`
- **Vector/Matrix**: `Cross`, `Dot`, `Determinant`, `Inverse`, `Transpose`
- **Memory**: `BufferRead`, `BufferWrite`, `Texture2dRead`
- **Atomic**: `AtomicExchange`, `AtomicFetchAdd`, etc.
- **Warp**: `WarpActiveSum`, `WarpPrefixSum`, etc.
- **Ray tracing**: `RayTracingTraceClosest`, `RayQueryCommitTriangle`
- **AD**: `RequiresGradient`, `Backward`, `PropagateGrad`

### C-Compatible Smart Pointers (`src/ffi.rs`)

```rust
pub struct CArc<T> {
    pub ptr: *mut T,
    pub inner: *mut CArcInner,
}

pub struct CBox<T> {
    pub ptr: *mut T,
    pub drop: extern "C" fn(*mut T),
}

pub struct CBoxedSlice<T> {
    pub ptr: *mut T,
    pub len: usize,
    pub drop: extern "C" fn(*mut T, usize),
}
```

---

## Key Algorithms and Transforms

### Transform Pipeline (`src/transform/mod.rs`)

| Transform | Purpose |
|-----------|---------|
| `ssa::ToSSA` | Converts imperative Update instructions to SSA form with Phi nodes |
| `autodiff::Autodiff` | Reverse-mode automatic differentiation |
| `fwd_autodiff::FwdAutodiff` | Forward-mode automatic differentiation |
| `dce::Dce` | Dead code elimination |
| `inliner::inline_callable` | Function inlining |
| `canonicalize_control_flow::CanonicalizeControlFlow` | Normalizes control flow |
| `ref2ret::Ref2Ret` | Converts reference returns to value returns |
| `reg2mem::Reg2Mem` | Register to memory conversion |

### Autodiff (`src/transform/autodiff.rs`)

Reverse-mode AD implementation:
- Forward sweep: Marks nodes requiring gradients via `RequiresGradient`
- Backward sweep: Accumulates gradients using chain rule

**Supported gradient rules**:
- Arithmetic: add, sub, mul, div, neg
- Vector: dot, cross, length, normalize
- Matrix: matmul, determinant, inverse, transpose
- Math: exp, log, sin, cos, sqrt, pow, trig functions
- Selection: min, max, select, clamp

### SSA Conversion (`src/transform/ssa.rs`)

- Promotes mutable `Local` variables to SSA values
- Handles `Update` instructions by tracking current value in `stored` map
- Inserts `Phi` nodes at merge points (if/then/else, loops)
- Supports GEP to `ExtractElement`/`InsertElement` conversion

### Dead Code Elimination (`src/transform/dce.rs`)

- Uses UseDef analysis to find unreachable nodes
- Removes pure computation nodes with no side effects
- Preserves memory operations and control flow

---

## CPU Backend Implementation

### Architecture (`cpu/mod.rs`)

- **Thread Pool**: Rayon-based parallel execution
- **Warp Size**: 1 (scalar execution)
- **Shader Compilation**: JIT via Clang/LLVM to shared library
- **Code Generation**: IR → C++ → LLVM IR → Machine code

### Code Generation (`cpu/codegen/cpp.rs`)

- Translates IR instructions to C++
- Handles vector types, matrices, textures, buffers
- Generates kernel entry point with proper argument marshaling
- Supports ray tracing via Embree

### Resource Management

- `BufferImpl`: Aligned host memory allocation
- `TextureImpl`: Mipmapped image storage
- `BindlessArrayImpl`: Array of buffer/texture descriptors
- `AccelImpl`: Embree scene acceleration structure

---

## Rust ↔ C++ FFI Integration

### C++ Header Generation

Both `luisa_compute_ir` and `luisa_compute_api_types` use `cbindgen`:

**`build.rs` workflow**:
1. Parse Rust source with `cbindgen`
2. Generate C++ headers (`ir.hpp`, `api_types.hpp`)
3. Generate C headers (`api_types.h`)
4. Write to `include/luisa/rust/`

### Static Library Linking

- `luisa_compute_ir_staticlib` builds as `staticlib`
- C++ links against `.a`/`.lib` file
- Uses `#[no_mangle]` extern "C" functions for C++ callable API

### Key FFI Functions

```rust
// Transform pipeline
luisa_compute_ir_transform_pipeline_new() -> *mut TransformPipeline
luisa_compute_ir_transform_pipeline_add_transform(pipeline, name)
luisa_compute_ir_transform_pipeline_transform(pipeline, module) -> Module
luisa_compute_ir_transform_auto(module) -> Module

// Library interface
luisa_compute_lib_interface() -> LibInterface
```

### API Types Sharing

- All types marked with `#[repr(C)]` for C layout compatibility
- Handle types are `u64` wrappers (`Buffer(u64)`, `Texture(u64)`)
- Callbacks use `extern "C" function pointers`
- `DeviceInterface` is a vtable struct of function pointers

---

## CMake Integration

**File**: `src/rust/CMakeLists.txt`

### Build Process

1. **Custom Command**: Invokes `cargo build` via CMake custom command
2. **Profile Selection**: `dev` for Debug, `release` for Release
3. **Feature Flags**: Passed based on CMake options (`cpu`, `remote`)

### Rust Artifacts

- Static libraries: `luisa_compute_ir_static`, `luisa_compute_api_types`
- Shared library: `luisa_compute_backend_impl`
- Platform-specific naming (`.lib`/`.a`, `.dll`/`.so`/`.dylib`)

### CMake Targets

- `luisa-compute-rust-meta` (INTERFACE): Links static Rust libs, Windows system libs
- `luisa_compute_backend_impl` (INTERFACE): Links shared Rust backend impl

### Platform Handling

- **macOS**: `install_name_tool` for rpath patching
- **Linux**: `patchelf` for rpath patching
- **Windows**: Copies `.dll`, `.lib`, and `.pdb` files

### Embree Integration

- Environment variable `EMBREE_ZIP_FILE` for custom Embree
- Downloads/builds Embree via Rust build script
- Copies embree DLLs to output directory

---

## Key Design Decisions

1. **IR Design**: Intrusive linked list of nodes with pool allocation for cache efficiency
2. **Type System**: Global type registry with structural equality via `context.rs`
3. **Memory Safety**: Custom `CArc` with C-compatible destructor callbacks
4. **Transform System**: Pipeline-based with modular passes (SSA, autodiff, DCE)
5. **CPU Backend**: JIT compilation to native code via LLVM/Clang
6. **C++ Interop**: Bidirectional via cbindgen (Rust→C++) and staticlib (C++→Rust)
