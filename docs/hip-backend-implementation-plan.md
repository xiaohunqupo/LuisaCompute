# HIP Backend Implementation Plan: XIR → LLVM IR

## Overview

Implement a full HIP backend for LuisaCompute using the XIR → LLVM IR → AMDGPU ISA compilation path. This leverages LLVM's AMDGPU backend directly without needing hipcc/NVRTC-style runtime compilation.

## Target Architecture

- **GPU**: AMD RX-9070XT (RDNA 4)
- **Target Triple**: `amdgcn-amd-amdhsa`
- **Target CPU**: `gfx1200` (hardcoded, runtime detection to be added later)
- **LLVM Components**: `core ir amdgpucodegen support irreader passes analysis`

---

## Branch

`feature/hip-llvm-codegen` (created from `next`)

---

## File Structure

```
src/backends/hip/
├── llvm_codegen/
│   ├── CMakeLists.txt
│   ├── hip_codegen_llvm.h           # Entry point header
│   ├── hip_codegen_llvm.cpp         # Entry point implementation
│   ├── hip_codegen_llvm_impl.h     # Main impl class
│   ├── hip_codegen_llvm_impl_type.cpp    # XIR → AMDGPU type mapping
│   ├── hip_codegen_llvm_impl_func.cpp    # Function/kernel translation
│   ├── hip_codegen_llvm_impl_inst.cpp     # Instruction dispatch
│   ├── hip_codegen_llvm_impl_cflow.cpp    # Control flow (if/loop/branch/phi)
│   ├── hip_codegen_llvm_impl_arith.cpp     # Arithmetic ops
│   ├── hip_codegen_llvm_impl_var.cpp       # Variable ops (alloca/load/store/gep)
│   ├── hip_codegen_llvm_impl_resource.cpp  # Resource ops (buffer/texture/bindless)
│   ├── hip_codegen_llvm_impl_cast.cpp      # Cast operations
│   └── hip_codegen_llvm_impl_atomic.cpp    # Atomic operations
├── hip_shader.h/cpp                  # Base shader class
├── hip_shader_native.h/cpp           # HIP compute shader (hipModuleLoadData, hipLaunchKernel)
├── hip_device.cpp                    # Modified: create_shader() implementation
└── hip_command_encoder.cpp           # Modified: ShaderDispatchCommand handler
```

---

## Commit History (14 Commits)

| # | Commit Message | Description |
|---|----------------|-------------|
| 1 | `hip-llvm: Create llvm_codegen directory structure` | CMakeLists.txt, entry point, AMDGPU target init |
| 2 | `hip-llvm: Implement type mapping` | XIR types → AMDGPU LLVM types (int/float/vector/matrix/buffer/texture) |
| 3 | `hip-llvm: Implement function translation` | Kernel/callable functions to LLVM |
| 4 | `hip-llvm: Implement instruction dispatch` | Switch table for instruction types |
| 5 | `hip-llvm: Implement control flow` | if/loop/branch, PHI handling |
| 6 | `hip-llvm: Implement arithmetic ops` | add/mul/sin/cos/sqrt/matmul, fast math |
| 7 | `hip-llvm: Implement variable ops` | alloca/load/store/gep, address spaces |
| 8 | `hip-llvm: Implement resource ops` | buffer/texture read/write, bindless array |
| 9 | `hip-llvm: Implement cast operations` | type conversions |
| 10 | `hip-llvm: Implement atomic operations` | atomicrmw, cmpxchg |
| 11 | `hip-llvm: Implement shader runtime` | Shader class, hipLaunchKernel dispatch |
| 12 | `hip-llvm: Hook into HIPDevice::create_shader` | Connect codegen to device |
| 13 | `hip-llvm: Implement command encoder dispatch` | ShaderDispatchCommand handler |
| 14 | `hip-llvm: Enable build and first test` | CMake updates, run `test_helloworld hip` |

---

## Key AMDGPU LLVM Intrinsics Mapping

| Category | CUDA (NVPTX) | AMDGPU |
|----------|--------------|--------|
| **Thread ID** | `llvm.nvvm.read.ptx.sreg.tid.x` | `llvm.amdgcn.workitem.id.x` |
| **Block ID** | `llvm.nvvm.read.ptx.sreg.ctaid.x` | `llvm.amdgcn.tgid.x` |
| **Block Size** | `llvm.nvvm.read.ptx.sreg.ntid.x` | `llvm.amdgcn.tgdim.x` |
| **Barrier** | `llvm.nvvm.barrier0` | `llvm.amdgcn.s.barrier` + `amdgcn.s_waitcnt` |
| **Warp Shuffle** | `llvm.nvvm.shfl.sync.bfly.i32` | `llvm.amdgcn.ds.bpermute` / `llvm.amdgcn.ds.perm` |
| **Texture Sample** | `llvm.nvvm.tex.2d.v4f32.f32` | `llvm.amdgcn.image.sample.2d.v4f32.f32` |
| **Texture Load** | `llvm.nvvm.suld.2d.i8.zero` | `llvm.amdgcn.image.load.2d.v4i32` |

---

## AMDGPU Address Spaces

| Address Space | NVPTX | AMDGPU |
|---------------|-------|--------|
| Global | 1 | 0 (flat) / 1 (global) |
| Shared | 3 | 3 (local/scratch) |
| Constant | 4 | 4 (constant) |
| Local | 5 | Not applicable |

---

## Important Caveats

### 1. Dynamic Parallelism - NOT SUPPORTED
Emit `LUISA_NOT_IMPLEMENTED()` if encountered. CUDA's dynamic parallelism (nested kernel launches) is deprecated.

### 2. Indirect Kernel Dispatch - NOT SUPPORTED
Emit `LUISA_NOT_IMPLEMENTED()` if encountered. Indirect dispatch buffers are deprecated.

### 3. No Device Runtime Library
Unlike CUDA's `cudadevrt`, AMDGPU/HIP doesn't have an equivalent device runtime library.

### 4. No Builtin Device Library
CUDA has `libdevice` (math intrinsics). HIP uses built-in `__ocml_*` functions or we generate inline.

### 5. Ray Tracing via HIPRT
HIPRT SDK is already linked in CMakeLists.txt. Acceleration structures (mesh/accel/curves) will use `hiprt*` APIs similar to how CUDA uses OptiX.

### 6. Sparse Resources
CUDA sparse APIs may have different AMD equivalents. Lower priority.

### 7. RX-9070XT Compatibility
`gfx1200` is RDNA 4 (very new). If compilation fails, may need to adjust target to `gfx1100` (RDNA 3) or `gfx1030` (RDNA 2).

---

## Test-Driven Development

After each commit, build incrementally:
```bash
cd cmake-build-debug
cmake --build . --target luisa-compute-backend-hip
```

When shader dispatch works (commit 13+):
```bash
./bin/test_helloworld hip
```

---

## CMakeLists.txt Modifications

### `src/backends/hip/CMakeLists.txt`

Add sources and link LLVM AMDGPU:

```cmake
# Add to LUISA_HIP_SOURCES:
hip_shader.cpp hip_shader.h
hip_shader_native.cpp hip_shader_native.h

# Link LLVM AMDGPU backend
luisa_compute_link_llvm_into_backend(hip
    COMPONENTS core ir amdgpucodegen support irreader passes analysis
    DEFINITIONS LUISA_COMPUTE_ENABLE_LLVM=1)
```

### `src/backends/hip/llvm_codegen/CMakeLists.txt` (new file)

```cmake
set(LUISA_HIP_LLVM_CODEGEN_SOURCES
    hip_codegen_llvm.cpp
    hip_codegen_llvm_impl_type.cpp
    hip_codegen_llvm_impl_func.cpp
    hip_codegen_llvm_impl_inst.cpp
    hip_codegen_llvm_impl_cflow.cpp
    hip_codegen_llvm_impl_arith.cpp
    hip_codegen_llvm_impl_var.cpp
    hip_codegen_llvm_impl_resource.cpp
    hip_codegen_llvm_impl_cast.cpp
    hip_codegen_llvm_impl_atomic.cpp
)
target_sources(luisa-compute-backend-hip PRIVATE ${LUISA_HIP_LLVM_CODEGEN_SOURCES})
```

---

## Implementation Notes

### Entry Point Signature

```cpp
// In hip_codegen_llvm.h
#include <luisa/xir/module.h>
#include <luisa/string.h>

namespace luisa::compute {

struct HIPCodegenLLVMConfig {
    uint sm_version;  // e.g., 1200 for gfx1200
    bool enable_fast_math;
    uint opt_level;   // 0-3
};

[[nodiscard]] luisa::string hip_codegen_llvm(
    const xir::Module &module,
    const HIPCodegenLLVMConfig &config) noexcept;

} // namespace luisa::compute
```

### Key Differences from CUDA

1. **Target Triple**: `amdgcn-amd-amdhsa` instead of `nvptx64-nvidia-cuda`
2. **CPU**: `gfx1200` instead of `sm_80`
3. **Barrier**: Need `llvm.amdgcn.s.barrier` + `llvm.amdgcn.s_waitcnt(vmcnt)`
4. **Wait Counter**: AMDGPU needs explicit `s_waitcnt` for memory ordering
5. **Image Intrinsics**: Use `llvm.amdgcn.image.*` instead of `llvm.nvvm.tex.*`

### Shader Launch

HIP uses `hipModuleLoadData` to load compiled code and `hipLaunchKernel` to dispatch:
```cpp
hipModuleLoadData(&module, compiled_code);
hipModuleGetFunction(&function, module, "kernel_main");
hipLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, stream, args);
```

---

## Progress Log

- [x] Plan created
- [ ] Commit 1: Create llvm_codegen directory structure
- [ ] Commit 2: Implement type mapping
- [ ] Commit 3: Implement function translation
- [ ] Commit 4: Implement instruction dispatch
- [ ] Commit 5: Implement control flow
- [ ] Commit 6: Implement arithmetic ops
- [ ] Commit 7: Implement variable ops
- [ ] Commit 8: Implement resource ops
- [ ] Commit 9: Implement cast operations
- [ ] Commit 10: Implement atomic operations
- [ ] Commit 11: Implement shader runtime
- [ ] Commit 12: Hook into HIPDevice::create_shader
- [ ] Commit 13: Implement command encoder dispatch
- [ ] Commit 14: Enable build and first test
