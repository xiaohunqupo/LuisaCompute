---
name: ir_pipeline
description: IR and XIR compiler pipeline, AST translation, SSA-based IR, instruction set, optimization passes, and control flow representation in LuisaCompute
---

# LuisaCompute IR and XIR Pipeline

## Overview

LuisaCompute has two intermediate representations:

| Aspect | IR (Legacy) | XIR (New/Preferred) |
|--------|-------------|---------------------|
| **Location** | `src/ir/`, `include/luisa/ir/` | `src/xir/`, `include/luisa/xir/` |
| **Implementation** | Rust-based (via `src/rust/`) | Pure C++ |
| **JSON Serialization** | `ast2json` → Rust IR | Native `xir2json`/`json2xir` |
| **Status** | Maintained for compatibility | Active development |
| **SSA Form** | Yes | Yes (with mem2reg pass) |
| **Basic Blocks** | Yes | Yes |

Both paths start from the same AST (`src/ast/`) and feed into backend codegen.

---

## Compiler Pipeline Flow

```
User C++ DSL / Python
         │
         ▼
┌─────────────────┐
│   DSL Tracing   │  src/dsl/ — captures C++ lambdas into AST
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│       AST       │  src/ast/ — expression/statement tree
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│  XIR   │ │   IR     │  src/xir/ (new) / src/ir/ (legacy)
│transl. │ │transl.   │
└───┬────┘ └────┬─────┘
    │           │
    ▼           ▼
┌─────────────────┐
│  Backend Codegen│  src/backends/<name>/
│  + Compilation  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPU Execution  │  src/runtime/
└─────────────────┘
```

**Rust IR path**: `src/rust/luisa_compute_ir/` performs additional transforms (autodiff, DCE, SSA, vectorize) before backend codegen.

**XIR path**: Pure C++ pipeline with `ast2xir` translator and XIR optimization passes.

---

## AST → IR Translation (Legacy)

**File**: `src/ir/ast2ir.cpp`

1. **AST → JSON**: Uses `to_json(function)` to serialize AST to JSON
2. **JSON → Rust IR**: Calls Rust FFI functions:
   - `ir::luisa_compute_ir_ast_json_to_ir_kernel()` for kernels
   - `ir::luisa_compute_ir_ast_json_to_ir_callable()` for callables
   - `ir::luisa_compute_ir_ast_json_to_ir_type()` for types

```
AST Function → JSON String → Rust IR Module → CArc<KernelModule/CallableModule>
```

**Key legacy IR classes** (defined in Rust, used via C FFI):
- `ir::KernelModule` / `ir::CallableModule`
- `ir::Node` — IR nodes (intrusive linked list)
- `ir::Instruction` — Local, Call, Phi, Loop, If, Switch, RayQuery, AdScope, etc.
- `ir::Type` — Type representation
- `ir::BasicBlock` — Basic block structure

---

## AST → XIR Translation

**File**: `src/xir/translators/ast2xir.cpp`

Direct C++ implementation using `XIRBuilder` API.

### Translation Context (`AST2XIRContext`)
- Maps AST variables to XIR Values
- Tracks break/continue target blocks
- Handles autodiff adjoint variables
- Manages constant/literal caching

### Expression Translation

| AST Expression | XIR Instruction |
|----------------|-----------------|
| `UnaryExpr` | `ArithmeticOp` (UNARY_MINUS, UNARY_BIT_NOT) |
| `BinaryExpr` | `ArithmeticOp` (BINARY_ADD, etc.) |
| `MemberExpr` | `EXTRACT`/`SHUFFLE` or `GEP` |
| `AccessExpr` | `GEP` + `LOAD` |
| `LiteralExpr` | `Constant` |
| `RefExpr` | Variable lookup or `SpecialRegister` |
| `CallExpr` | Various opcodes based on `CallOp` |
| `CastExpr` | `CastInst` (STATIC_CAST, BITWISE_CAST) |

### Statement Translation

| AST Statement | XIR Control Flow |
|---------------|------------------|
| `IfStmt` | `IfInst` + true/false/merge blocks |
| `SwitchStmt` | `SwitchInst` + case/default/merge blocks |
| `ForStmt` | `LoopInst` (prepare/body/update/merge) |
| `LoopStmt` | `SimpleLoopInst` (do-while style) |
| `BreakStmt` | `BreakInst` |
| `ContinueStmt` | `ContinueInst` |
| `ReturnStmt` | `ReturnInst` |
| `AutoDiffStmt` | `AutodiffScopeInst` |

---

## XIR Core Architecture

### Value Hierarchy

```
Value (base class)
├── GlobalValue (module-scoped)
│   ├── Function
│   │   ├── KernelFunction (entry point with block size)
│   │   ├── CallableFunction (user-defined functions)
│   │   └── ExternalFunction (external linkage)
│   ├── Constant (literal values)
│   ├── Undefined (placeholder for undef)
│   └── SpecialRegister (builtin variables)
│       ├── SPR_ThreadID, SPR_BlockID, SPR_DispatchID, etc.
├── FunctionScopeValue (function-scoped)
│   └── BasicBlock (container for instructions)
└── BlockScopeValue (block-scoped)
    └── Instruction (computation)
        ├── TerminatorInstruction (control flow)
        │   ├── BranchInst
        │   ├── ConditionalBranchInst
        │   ├── IfInst, SwitchInst, LoopInst, etc.
        │   └── ReturnInst
        └── Non-terminator instructions
```

### Key Classes

#### Module (`include/luisa/xir/module.h`)
- Container for all global values
- Creates/manages functions, constants, undefined values, special registers
- Uniquifies constants via hash table

#### Function (`include/luisa/xir/function.h`)
- `ArgumentList` — function parameters
- `BasicBlockList` — CFG nodes
- `FunctionDefinition` — for kernels/callables with body blocks
- Traversal: `traverse_basic_blocks()`, `traverse_instructions()`

#### BasicBlock (`include/luisa/xir/basic_block.h`)
- `InstructionList` — sequential instructions
- `is_terminated()`, `terminator()` — control flow queries
- `traverse_predecessors()`, `traverse_successors()` — CFG navigation

#### Instruction (`include/luisa/xir/instruction.h`)
- Derived via `DerivedInstruction<>` template
- `is_terminator()` — ends basic block
- `control_flow_merge()` — for structured control flow
- `clone()` — for pass transformations

#### Value & Use (`include/luisa/xir/value.h`, `use.h`)
- **SSA Form**: Each value has a `UseList` tracking all users
- `replace_all_uses_with()` — for substitution
- `is_lvalue()` — for memory operations (alloca, gep results)

---

## XIR Instruction Set

### Control Flow Instructions

| Instruction | Purpose | Blocks |
|-------------|---------|--------|
| `IfInst` | Conditional branch | true, false, merge |
| `SwitchInst` | Multi-way branch | cases, default, merge |
| `LoopInst` | For/while loops | prepare, body, update, merge |
| `SimpleLoopInst` | Do-while loops | body, merge |
| `BranchInst` | Unconditional jump | target |
| `ConditionalBranchInst` | Branches on condition | true, false |
| `ReturnInst` | Function return | — |
| `BreakInst` / `ContinueInst` | Loop control | target |
| `UnreachableInst` | Unreachable code | — |

### Memory Instructions

| Instruction | Purpose |
|-------------|---------|
| `AllocaInst` | Stack allocation (LOCAL/SHARED) |
| `LoadInst` | Read from memory |
| `StoreInst` | Write to memory |
| `GEPInst` | Get element pointer (indexing) |

### SSA Instructions

| Instruction | Purpose |
|-------------|---------|
| `PhiInst` | Phi node for SSA (block, value) pairs |

### Arithmetic Instructions (`ArithmeticOp`)

**Unary**: `UNARY_MINUS`, `UNARY_BIT_NOT`

**Binary**: `BINARY_ADD`, `SUB`, `MUL`, `DIV`, `MOD`, `BIT_AND`, `BIT_OR`, `BIT_XOR`, `SHIFT_LEFT`, `SHIFT_RIGHT`, `ROTATE_LEFT`, `ROTATE_RIGHT`, comparison ops

**Math**: `ABS`, `MIN`, `MAX`, `CLAMP`, `SATURATE`, `LERP`, `SMOOTHSTEP`, trigonometric, exponential, logarithmic

**Vector/Matrix**: `DOT`, `CROSS`, `NORMALIZE`, `MATRIX_COMP_MUL`, `MATRIX_LINALG_MUL`, `MATRIX_DETERMINANT`, `MATRIX_TRANSPOSE`, `MATRIX_INVERSE`

**Aggregate**: `AGGREGATE`, `SHUFFLE`, `EXTRACT`, `INSERT`

### Resource Instructions

| Category | Operations |
|----------|-----------|
| `ResourceQueryOp` | Buffer/texture size, bindless queries, ray tracing queries |
| `ResourceReadOp` | Buffer read, texture read, bindless read |
| `ResourceWriteOp` | Buffer write, texture write, ray tracing updates |

### Atomic Instructions (`AtomicOp`)

`EXCHANGE`, `COMPARE_EXCHANGE`, `FETCH_ADD`, `FETCH_SUB`, `FETCH_AND`, `FETCH_OR`, `FETCH_XOR`, `FETCH_MIN`, `FETCH_MAX`

### Thread Group Instructions (`ThreadGroupOp`)

Warp operations, block synchronization, shader execution reordering, raster quad derivatives

### Ray Query Instructions

`RayQueryLoopInst`, `RayQueryDispatchInst`, `RayQueryObjectReadInst`, `RayQueryObjectWriteInst`, `RayQueryPipelineInst`

### Autodiff Instructions

`AutodiffScopeInst`, `AutodiffIntrinsicInst` (requires_gradient, gradient, accumulate_gradient, backward, detach)

---

## XIR Optimization Passes

**Location**: `src/xir/passes/`

### Core Passes

| Pass | File | Purpose |
|------|------|---------|
| **DCE** | `dce.cpp` | Removes unused instructions, unreachable blocks, dead allocas; evaluates static branches |
| **Mem2Reg** | `mem2reg.cpp` | Promotes allocas to SSA registers using dominance tree and frontiers; inserts PHI nodes |
| **Dominance Tree** | `dom_tree.cpp` | Cooper et al. 2001 algorithm; computes immediate dominators and dominance frontiers |
| **Early Return Elimination** | `early_return_elimination.cpp` | Converts early returns to structured control flow |

### Analysis Passes

| Pass | File | Purpose |
|------|------|---------|
| **Call Graph** | `call_graph.cpp` | Call graph analysis |
| **Pointer Usage** | `pointer_usage.cpp` | Pointer usage analysis |
| **Lexical Scope** | `lex_scope_analysis.cpp` | Lexical scope analysis |
| **Aggregate Field Bitmask** | `aggregate_field_bitmask.cpp` | Aggregate field usage analysis |

### Transformation Passes

| Pass | File | Purpose |
|------|------|---------|
| **SROA** | `sroa.cpp` | Scalar replacement of aggregates |
| **Outline** | `outline.cpp` | Function outlining |
| **Autodiff** | `autodiff.cpp` | Autodiff transformations |
| **Lower Ray Query** | `lower_ray_query_loop.cpp` | Ray query lowering |
| **Reg2Mem** | `reg2mem.cpp` | Register to memory conversion |
| **Promote Ref Arg** | `promote_ref_arg.cpp` | Reference argument promotion |
| **Transpose GEP** | `transpose_gep.cpp` | Transpose GEP through loads/stores |
| **Trace GEP** | `trace_gep.cpp` | GEP analysis/tracing |
| **Local Load Elimination** | `local_load_elimination.cpp` | Redundant load elimination |
| **Local Store Forward** | `local_store_forward.cpp` | Store-to-load forwarding |
| **Unused Callable Removal** | `unused_callable_removal.cpp` | Dead function elimination |

---

## Control Flow Representation

XIR uses **structured control flow** with explicit merge blocks:

```
IfInst:
  - condition: Value*
  - true_block: BasicBlock*
  - false_block: BasicBlock*
  - merge_block: BasicBlock*

LoopInst:
  - prepare_block: BasicBlock*
  - body_block: BasicBlock*
  - update_block: BasicBlock*
  - merge_block: BasicBlock*
```

This design:
- Maintains SSA property
- Enables structured transformations
- Maps well to GPU shader control flow requirements
- Supports PHI nodes at merge points

---

## Metadata System

**File**: `include/luisa/xir/metadata.h`

Metadata types (`DerivedMetadataTag`):
- `NAME` — Symbol names
- `LOCATION` — Source file/line information
- `COMMENT` — Debug comments
- `CURVE_BASIS` — Ray tracing curve basis

Applied via `MetadataListMixin` to Values and Instructions.

---

## JSON Serialization

### IR (Legacy)
- Uses `ast2json` for AST serialization
- Rust IR can be serialized/deserialized
- C API: `luisa_compute_ir_ast_json_to_ir_*`

### XIR
**Files**: `src/xir/translators/xir2json.cpp`, `json2xir.cpp`

Uses yyjson library for module serialization, cross-process communication, and debugging.

---

## Key Design Patterns

1. **Intrusive Lists**: `ManagedIntrusiveList` for efficient node management
2. **CRTP**: `DerivedValue<>`, `DerivedInstruction<>`, `DerivedFunction<>`
3. **Visitor Pattern**: `traverse_basic_blocks()`, `traverse_instructions()`
4. **Builder Pattern**: `XIRBuilder` for instruction creation
5. **Mixin Pattern**: `MetadataListMixin`, `ControlFlowMergeMixin`, `InstructionOpMixin`
6. **Use-Def Chains**: `Use` objects track value users for SSA

---

## Adding a New XIR Pass

1. Create pass implementation in `src/xir/passes/<name>.cpp`
2. Create header in `include/luisa/xir/passes/<name>.h`
3. Register in `src/xir/CMakeLists.txt`
4. Follow existing pass conventions:
   - Accept `Module &` or `Function &`
   - Return `bool` indicating whether changes were made
   - Use `XIRBuilder` for instruction insertion
   - Call `replace_all_uses_with()` for value substitution
