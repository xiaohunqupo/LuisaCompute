# Vulkan 后端 Motion Blur 实现总结

## 概述

基于 `VK_NV_ray_tracing_motion_blur` 扩展，本次实现在 LuisaCompute 的 Vulkan 后端中完整添加了光线追踪运动模糊（Ray Tracing Motion Blur）支持。实现涵盖加速结构构建（BLAS/TLAS）、光线追踪管线、SPIR-V 后处理、HLSL 代码生成等多个层次。

## 一、整体架构

实现采用 **双路径架构**：

| 路径 | 适用场景 | 执行模型 |
|------|---------|---------|
| Compute + Inline Ray Query | 不使用 motion blur 的 shader | `RayQuery<>::TraceRayInline()` |
| Ray Tracing Pipeline | 使用 motion blur 的 shader | `TraceRay()` + `OpTraceRayMotionNV` |

核心思路：当 HLSL codegen 检测到 `RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR` 调用时，走 ray tracing pipeline 路径，通过 SPIR-V 后处理将 `OpTraceRayKHR` 替换为 `OpTraceRayMotionNV`。

## 二、数据模型层

### 2.1 PrimitiveBase 类型系统

**新增文件**: `src/backends/vk/primitive_base.h`

引入 `PrimitiveBase` 基类，继承自 `Resource`，为 TLAS 提供区分静态几何和运动实例的能力：

```cpp
class PrimitiveBase : public Resource {
public:
    enum class PrimTag { BLAS, MOTION_INSTANCE };
    [[nodiscard]] auto prim_tag() const noexcept;
    [[nodiscard]] bool is_motion_instance() const noexcept;
};
```

**改动**: `Blas` 的继承关系从 `Resource` 改为 `PrimitiveBase`，构造时传入 `PrimTag::BLAS`。

### 2.2 MotionInstance 类

**新增文件**: `src/backends/vk/motion_instance.h`, `src/backends/vk/motion_instance.cpp`

`MotionInstance` 继承自 `PrimitiveBase`，封装运动实例的 CPU 端数据：

| 成员 | 类型 | 说明 |
|------|------|------|
| `_option` | `AccelMotionOption` | 运动模式（SRT/Matrix）、关键帧数、时间范围 |
| `_child` | `Blas*` | 子 BLAS 指针 |
| `_keyframes` | `vector<MotionInstanceTransform>` | 关键帧变换数据 |

构造函数校验：
- 设备必须支持 motion blur（`device->enable_motion_blur()`）
- 关键帧数 ≥ 2

初始化时将 keyframes 填充为单位变换（SRT 模式用 `MotionInstanceTransformSRT{}`，Matrix 模式用 `make_float4x4(1.f)`）。

## 三、设备层

### 3.1 扩展检测与启用

**修改文件**: `src/backends/vk/device.cpp`, `src/backends/vk/device.h`

设备创建时检测并启用以下扩展：
- `VK_NV_ray_tracing_motion_blur`（运动模糊）
- `VK_KHR_ray_tracing_pipeline`（光线追踪管线，motion blur 的前置依赖）

同时启用对应的 device features：
- `VkPhysicalDeviceRayTracingMotionBlurFeaturesNV::rayTracingMotionBlur = VK_TRUE`
- `VkPhysicalDeviceRayTracingPipelineFeaturesKHR::rayTracingPipeline = VK_TRUE`

`device.h` 新增 `motion_blur_enabled` 位域和 `enable_motion_blur()` 访问器，以及 `set_accel_motion_kernel` LazyLoadShader。

### 3.2 设备接口实现

`device.cpp` 实现了 `create_motion_instance()` 和 `destroy_motion_instance()` 设备接口方法，与 CUDA 后端保持一致的 API。

## 四、加速结构构建

### 4.1 BLAS 层面：顶点运动模糊

**修改文件**: `src/backends/vk/blas.cpp`, `src/backends/vk/blas.h`

`Blas` 新增 `has_motion()` 方法，检查 `_option.motion.is_enabled()`。

**Mesh 构建路径改造**（`blas.cpp:103-161`）：

1. 检测 `_option.motion.is_enabled()` 后：
   - 计算每关键帧顶点数：`vertex_count_per_keyframe = total_vertex_count / keyframe_count`
   - 当 `keyframe_count == 2` 时，使用 `VkAccelerationStructureGeometryMotionTrianglesDataNV` 提供第二帧顶点数据
   - `triangles.vertexData` 指向关键帧 0，`motion_triangles->vertexData` 指向关键帧 1（偏移 `keyframe_size_bytes`）
   - `triangles.pNext` 链接到 motion triangles 结构

2. 设置 motion build flags：
   - `VkAccelerationStructureBuildGeometryInfoKHR::flags |= VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV`
   - `VkAccelerationStructureCreateInfoKHR::createFlags |= VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV`

**限制**: `VK_NV_ray_tracing_motion_blur` 仅支持恰好 2 个关键帧。当 `keyframe_count > 2` 时，走非运动路径（退化处理）。

### 4.2 TLAS 层面：Motion Instance 构建

**修改文件**: `src/backends/vk/tlas.cpp`, `src/backends/vk/tlas.h`

#### 4.2.1 Motion 检测

`Tlas` 新增 `_has_motion` 成员，在 `pre_build()` 中通过两步检测：

1. **预扫描**：遍历 modifications，检查是否存在 `MotionInstance` 或子 BLAS 启用了 motion
2. **构建时检查**：在 resolved_meshes 循环中，再次检查 `resolved_meshes[idx]->has_motion()`

#### 4.2.2 Instance Buffer 布局切换

当 `_has_motion == true` 时，TLAS 使用两个 instance buffer：

| 属性 | 非运动模式 | 运动模式（TLAS 构建用） | 运动模式（Shader 读取用） |
|------|-----------|----------------------|------------------------|
| 实例结构体 | `VkAccelerationStructureInstanceKHR` | `VkAccelerationStructureMotionInstanceNV` | `VkAccelerationStructureInstanceKHR` |
| 步长 | 64 字节 | 160 字节（16 字节对齐） | 64 字节 |
| 缓冲区 | `_instance_buffer` | `_motion_instance_buffer` | `_instance_buffer` |
| 构建方式 | Compute Shader（accel_process_vk） | CPU 直接填充（upload buffer） | CPU 直接填充（upload buffer） |

> **注意**：`VkAccelerationStructureMotionInstanceNV` 的 C struct 大小为 152 字节，包含：
> - `type` (4 bytes) + `flags` (4 bytes) = 8 bytes 头部
> - `data` union (144 bytes)：最大成员为 `VkAccelerationStructureSRTMotionInstanceNV` (2×64 + 4 + 4 + 8 = 144 bytes)
>
> 但 Vulkan 驱动要求 motion instance 在 instance buffer 中 16 字节对齐，因此实际步长为
> `ceil(152/16)*16 = 160` 字节。使用 152 字节步长会导致多实例场景下只有第一个实例被正确识别。
>
> HLSL 中 `RWByteAddressBuffer` 的所有操作必须 4 字节对齐，这是 HLSL ByteAddressBuffer 的标准要求，与 Vulkan 无关。
>
> 运动模式下需要维护两个 buffer 的原因：Shader 通过 `StructuredBuffer<_MeshInst>`（64 字节步长）
> 读取 instance 数据（如 `instance_user_id`、`instance_transform` 等），而 TLAS 构建需要
> 160 字节步长的 motion instance buffer。两者步长不同，无法共用同一个 buffer。

#### 4.2.3 运动模式 Instance Buffer 填充

运动模式下不使用 compute shader，而是直接在 CPU 端填充 upload buffer（`tlas.cpp:147-216`）：

```
偏移量   大小        字段
0        4 bytes     type = STATIC_NV (0)
4        4 bytes     flags = 0
8        48 bytes    transform matrix[3][4] (VkTransformMatrixKHR)
56       4 bytes     instanceCustomIndex:24 | mask:8
60       4 bytes     SBT offset:24 | geometry flags:8
64       8 bytes     accelerationStructureReference (uint64)
72       112 bytes   填充（union 最大为 SRT motion 176 bytes）
```

所有实例当前均写入 `type=0`（STATIC），即 motion instance buffer 中的静态实例变体。

#### 4.2.4 Build Flags

```cpp
// Build geometry info
if (_has_motion) {
    _acceleration_build_geometry_info->flags |= VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV;
}

// Create info
VkAccelerationStructureMotionInfoNV motion_info{};
motion_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MOTION_INFO_NV;
motion_info.maxInstances = instance_count;
if (_has_motion) {
    acceleration_structure_create_info.createFlags = VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV;
    acceleration_structure_create_info.pNext = &motion_info;
}
```

#### 4.2.5 非运动路径保留

当 `_has_motion == false` 时，保留原有的 compute shader 路径（`accel_process_vk`），通过 GPU dispatch 填充 64 字节的 `VkAccelerationStructureInstanceKHR`。

### 4.3 Motion 版 accel_process Compute Shader

**新增文件**: `src/backends/common/hlsl/builtin/accel_process_vk_motion`
**注册**: `src/backends/vk/builtin_kernel.cpp` → `load_accel_motion_set_kernel()`

这是一个 HLSL compute shader，输出 `RWByteAddressBuffer`（按 160 字节步长手动寻址），写入 STATIC 类型的 motion instance：

```hlsl
// 关键逻辑
static const uint MOTION_INST_STRIDE = 160u;  // sizeof(VkAccelerationStructureMotionInstanceNV)=152, 对齐到 16 字节 = 160
static const uint STATIC_DATA_OFFSET = 8u;    // type(4) + flags(4)

uint base = v.index * MOTION_INST_STRIDE;
_InstBuffer.Store(base, 0u);              // type = STATIC
_InstBuffer.Store(base + 4u, 0u);         // flags = 0
uint data_base = base + STATIC_DATA_OFFSET;
// ... 写入 transform, instanceID, mask, SBT flags, accel reference
```

使用 `RWByteAddressBuffer` 而非 `RWStructuredBuffer` 的原因：
- `RWByteAddressBuffer` 允许完全灵活的字节寻址，适合非标准布局的结构体
- HLSL 的 `RWByteAddressBuffer` 所有操作必须 4 字节对齐（这是 HLSL ByteAddressBuffer 的标准要求，与 Vulkan 无关）
- `StructuredBuffer<T>` 要求元素类型具有固定布局，而 160 字节的 motion instance 在 HLSL 中难以精确映射
- 注意：`StructuredBuffer<uint16_t>` 等小类型是支持的，但复杂嵌套结构体在 HLSL 中的布局可能不符合预期

> **重要**：`VkAccelerationStructureMotionInstanceNV` 的 C struct 大小为 152 字节，但 Vulkan 驱动要求
> motion instance 在 instance buffer 中 16 字节对齐。因此实际步长为 `ceil(152/16)*16 = 160` 字节。
> 使用 `sizeof(VkAccelerationStructureMotionInstanceNV)` (152) 作为步长会导致多实例 TLAS 构建时
> 只有第一个实例被正确识别。

## 五、光线追踪管线

### 5.1 RayTracingShader 类

**新增文件**: `src/backends/vk/rt_shader.h`, `src/backends/vk/rt_shader.cpp`

全新的 shader 类型，继承自 `Shader`，管理完整的 ray tracing pipeline：

#### 5.1.1 管线创建

```cpp
// 3 个 shader stage
stages[0] = RayGen  ("main_raygen")
stages[1] = Miss    ("main_miss")
stages[2] = ClosestHit ("main_closesthit")

// 3 个 shader group
groups[0] = GENERAL (raygen)
groups[1] = GENERAL (miss)
groups[2] = TRIANGLES_HIT_GROUP (closesthit)

// 关键 flag
rt_pipeline_ci.flags = VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV;
```

#### 5.1.2 Shader Binding Table (SBT)

SBT 布局：
```
| RayGen Region | Miss Region | Hit Region |
| aligned_handle | aligned_handle | aligned_handle |
```

使用 VMA 分配 `CPU_TO_GPU` 内存，通过 `vkGetRayTracingShaderGroupHandlesKHR` 获取 handle 并写入 SBT buffer。

三个 `VkStridedDeviceAddressRegionKHR` 区域供 `vkCmdTraceRaysKHR` 使用。

#### 5.1.3 编译流程

```
RayTracingShader::compile()
  ├── CodegenUtility::RayTracingCodegen() → HLSL 代码
  ├── ShaderCompiler::compile_raytracing() → SPIR-V (lib_6_5)
  ├── patch_spirv_for_motion_blur() → 后处理 SPIR-V
  └── new RayTracingShader() → 创建 VkPipeline
```

### 5.2 Shader 类型系统扩展

**修改文件**: `src/backends/vk/shader.h`, `src/backends/vk/shader.cpp`

`Shader::ShaderTag` 新增 `kRayTracingShader` 枚举值。

Shader 构造函数中根据 tag 设置对应的 `VkShaderStageFlagBits`：
```cpp
case ShaderTag::kRayTracingShader:
    return VK_SHADER_STAGE_RAYGEN_BIT_KHR |
           VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
           VK_SHADER_STAGE_MISS_BIT_KHR;
```

## 六、SPIR-V 后处理

**新增文件**: `src/backends/vk/spirv_motion_patch.h`, `src/backends/vk/spirv_motion_patch.cpp`

这是整个实现中最精巧的部分。由于 DXC 无法直接生成 `SPV_NV_ray_tracing_motion_blur` 指令，采用 **编译后 patch** 方案。

### 6.1 工作原理

DXC 编译 HLSL 时，`TraceRay()` 会生成 `OpTraceRayKHR` 指令。Patcher 将其替换为 `OpTraceRayMotionNV`，同时注入所需的 capability 和 extension。

### 6.2 三阶段处理

**Phase 1: 扫描** — 遍历 SPIR-V 二进制，统计 `OpTraceRayKHR` 数量

**Phase 2: 建立映射** — 构建 `result ID → instruction offset` 映射表，用于查找 `OpCompositeConstruct`

**Phase 3: 提取 time 值** — 对每个 `OpTraceRayKHR`：
1. 获取 payload 变量 ID（最后一个操作数）
2. 反向扫描找到 `OpStore` 到该 payload 的指令
3. 通过 `OpCompositeConstruct` 找到 payload 的构造来源
4. 提取 `OpCompositeConstruct` 的最后一个元素（即 time 值）

**Phase 4: 重建 SPIR-V**
1. 在 Capability 段末尾插入 `OpCapability RayTracingMotionBlurNV (5341)`
2. 在 Extension 段末尾插入 `OpExtension "SPV_NV_ray_tracing_motion_blur"`
3. 将 `OpTraceRayKHR`（12 words）替换为 `OpTraceRayMotionNV`（13 words），在 rayTmax 和 payload 之间插入 time 操作数

### 6.3 SPIR-V 指令格式对比

```
OpTraceRayKHR:       Accel, RayFlags, CullMask, SBTOffset, SBTStride, MissIndex, RayDesc, Payload
OpTraceRayMotionNV:  Accel, RayFlags, CullMask, SBTOffset, SBTStride, MissIndex, RayDesc, Time, Payload
```

### 6.4 关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `SpvOpTraceRayKHR` | 4445 | 标准 trace 指令 |
| `SpvOpTraceRayMotionNV` | 5339 | Motion blur trace 指令 |
| `SpvCapabilityRayTracingMotionBlurNV` | 5341 | Motion blur capability |

## 七、HLSL Codegen 层

### 7.1 Motion Trace 调用

**修改文件**: `src/backends/common/hlsl/codegen_utils/function_codegen.cpp`

`RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR` 不再降级为非 motion 版本，而是生成：

```cpp
str << "_TraceClosestMotion("sv;
args[0]->accept(vis); // accel
str << ',';
args[1]->accept(vis); // ray
str << ',';
args[2]->accept(vis); // time ← 关键：不再忽略
str << ',';
args[3]->accept(vis); // mask
str << ')';
```

`RAY_TRACING_QUERY_ALL_MOTION_BLUR` 和 `RAY_TRACING_QUERY_ANY_MOTION_BLUR` 仍然降级为非 motion 版本（丢弃 time 参数），因为 inline ray query 不支持 time。

### 7.2 Ray Tracing Codegen 入口

**修改文件**: `src/backends/common/hlsl/codegen_utils/entry_points.cpp`

新增 `CodegenUtility::RayTracingCodegen()` 方法，与 `Codegen()` 类似但：
- 设置 `opt->isRayTracing = true`
- 额外包含 `raytracing_motion_header.bytes`

### 7.3 CodegenStackData 扩展

**修改文件**: `src/backends/common/hlsl/codegen_stack_data.h`

新增 `bool isRayTracing : 1 = false` 标志，用于区分 codegen 上下文。

### 7.4 Ray Tracing Shader 编译器

**修改文件**: `src/backends/common/hlsl/shader_compiler.h`, `src/backends/common/hlsl/shader_compiler.cpp`

新增 `compile_raytracing()` 方法：
- Shader target: `lib_6_5`（library profile，支持 raygen/miss/closesthit 入口）
- SPIR-V target: `vulkan1.2`
- 扩展: `-fspv-extension=SPV_KHR_ray_tracing`

### 7.5 Motion Header HLSL

**新增文件**: `src/backends/common/hlsl/builtin/raytracing_motion_header.bytes`

定义 motion blur ray tracing 的 HLSL 基础设施：

```hlsl
struct _MotionPayload {
    uint inst_idx;   // 实例索引 (0xFFFFFFFF = miss)
    uint prim_idx;   // 图元索引
    float2 bary;     // 重心坐标
    float ray_t;     // 光线 T 值
    float time;      // ← 关键：motion blur 时间参数
};

[shader("miss")] void main_miss(inout _MotionPayload payload);
[shader("closesthit")] void main_closesthit(inout _MotionPayload payload, BuiltInTriangleIntersectionAttributes attr);

template<typename T>
_Hit1 _TraceClosestMotion(RaytracingAccelerationStructure accel, T rayDesc, float time, uint mask);
```

`_TraceClosestMotion` 的实现：
1. 将 time 存入 `_MotionPayload.time` 字段
2. 调用标准 `TraceRay()`（DXC 生成 `OpTraceRayKHR`）
3. SPIR-V patcher 后续会找到这个 time 值并替换指令

## 八、命令分发与重排序

### 8.1 Stream 命令处理

**修改文件**: `src/backends/vk/stream.cpp`

在命令处理循环中，`EMotionInstanceBuildCommand` 提前处理（在重排序/预处理之前）：

```cpp
case Command::Tag::EMotionInstanceBuildCommand: {
    auto c = static_cast<MotionInstanceBuildCommand const *>(command.get());
    auto mi = reinterpret_cast<MotionInstance *>(c->handle());
    mi->set_child(reinterpret_cast<Blas *>(c->child()));
    mi->set_keyframes(const_cast<MotionInstanceBuildCommand *>(c)->steal_keyframes());
} break;
```

### 8.2 命令重排序

**修改文件**: `src/backends/common/command_reorder_visitor.h`

新增 `visit(const MotionInstanceBuildCommand *)` 重载：

```cpp
void visit(const MotionInstanceBuildCommand *command) noexcept override {
    add_command(command, set_write(command->handle(), Range(), ResourceType::Accel));
}
```

确保 `MotionInstanceBuildCommand` 在 `AccelBuildCommand` 之前执行（将 motion instance handle 注册为 Accel 类型的写操作）。

## 九、Builtin Kernel 注册

**修改文件**: `src/backends/vk/builtin_kernel.h`, `src/backends/vk/builtin_kernel.cpp`

- `BuiltinKernel::load_accel_motion_set_kernel()` — 加载 motion 版 instance buffer 填充 kernel
- `builtin_kernel.h` 新增静态方法声明

## 十、编译配置

### 10.1 CMakeLists 变更

**修改文件**:
- `src/backends/common/hlsl/CMakeLists.txt` — 添加新 builtin 文件
- `src/backends/common/hlsl/builtin/hlsl_builtin.hpp` — 嵌入新资源
- `src/tests/CMakeLists.txt` — 添加 motion blur 测试

### 10.2 新文件清单

| 文件 | 用途 |
|------|------|
| `src/backends/vk/primitive_base.h` | 基类：区分 BLAS / MotionInstance |
| `src/backends/vk/motion_instance.h` | MotionInstance 头文件 |
| `src/backends/vk/motion_instance.cpp` | MotionInstance 实现 |
| `src/backends/vk/rt_shader.h` | Ray Tracing 管线 shader 头文件 |
| `src/backends/vk/rt_shader.cpp` | Ray Tracing 管线实现（含 SBT） |
| `src/backends/vk/spirv_motion_patch.h` | SPIR-V 后处理头文件 |
| `src/backends/vk/spirv_motion_patch.cpp` | SPIR-V patcher 实现 |
| `src/backends/common/hlsl/builtin/raytracing_motion_header.bytes` | Motion RT HLSL 头 |
| `src/backends/common/hlsl/builtin/accel_process_vk_motion` | Motion instance buffer 填充 shader |

### 10.3 修改文件清单

| 文件 | 改动内容 |
|------|---------|
| `src/backends/vk/device.cpp` | 扩展检测、feature chain、motion blur 启用 |
| `src/backends/vk/device.h` | motion_blur_enabled、set_accel_motion_kernel |
| `src/backends/vk/blas.h` | has_motion() 方法 |
| `src/backends/vk/blas.cpp` | BLAS 顶点 motion 构建、motion flags |
| `src/backends/vk/tlas.h` | _has_motion 成员 |
| `src/backends/vk/tlas.cpp` | TLAS motion instance buffer、build flags |
| `src/backends/vk/shader.h` | kRayTracingShader tag |
| `src/backends/vk/shader.cpp` | RT shader stage bits |
| `src/backends/vk/stream.cpp` | MotionInstanceBuildCommand 处理 |
| `src/backends/vk/builtin_kernel.h` | load_accel_motion_set_kernel 声明 |
| `src/backends/vk/builtin_kernel.cpp` | motion kernel 注册 |
| `src/backends/common/hlsl/function_codegen.cpp` | Motion blur trace codegen |
| `src/backends/common/hlsl/entry_points.cpp` | RayTracingCodegen 入口 |
| `src/backends/common/hlsl/shader_compiler.h` | compile_raytracing 声明 |
| `src/backends/common/hlsl/shader_compiler.cpp` | compile_raytracing 实现 |
| `src/backends/common/hlsl/hlsl_codegen.h` | RayTracingCodegen 声明 |
| `src/backends/common/hlsl/codegen_stack_data.h` | isRayTracing 标志 |
| `src/backends/common/command_reorder_visitor.h` | MotionInstanceBuildCommand 重排序 |

## 十一、数据流总览

```
用户调用 accel.intersect_motion(ray, time, options)
        │
        ▼
AST: CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR
        │
        ▼
HLSL Codegen: _TraceClosestMotion(accel, ray, time, mask)
        │
        ▼
raytracing_motion_header.bytes: TraceRay(accel, ..., payload)
  payload.time = time
        │
        ▼
DXC 编译: lib_6_5 → SPIR-V (OpTraceRayKHR)
        │
        ▼
SPIR-V Patcher:
  1. 找到 OpCompositeConstruct 中的 time 值
  2. OpTraceRayKHR → OpTraceRayMotionNV
  3. 注入 Capability + Extension
        │
        ▼
RayTracingShader::compile():
  ├── 创建 VkRayTracingPipelineCreateInfoKHR
  │     flags = ALLOW_MOTION_BIT_NV
  ├── 构建 SBT (RayGen + Miss + ClosestHit)
  └── vkCreateRayTracingPipelinesKHR
        │
        ▼
运行时:
  ├── TLAS 构建 (motion instance buffer, MOTION_BIT_NV)
  ├── BLAS 构建 (motion triangles, MOTION_BIT_NV)
  └── vkCmdTraceRaysKHR 发射光线
        │
        ▼
Vulkan 驱动:
  └── 硬件根据 time 参数在关键帧之间插值
```

## 十二、与 CUDA 后端的对比

| 方面 | CUDA/OptiX | Vulkan |
|------|-----------|--------|
| Motion Transform | 独立 traversable handle（指针型） | 内嵌在 instance buffer 中（184B） |
| 关键帧支持 | N 个关键帧 | 仅 2 个关键帧（NV 扩展限制） |
| Time 传递 | PTX 内联汇编直接传参 | SPIR-V 后处理 patch |
| 执行模型 | Inline ray query + time 参数 | Ray Tracing Pipeline + OpTraceRayMotionNV |
| SRT 插值 | OptiX RT Core 硬件 slerp | Vulkan 驱动层处理 |
| 架构变更 | 无需改变执行模型 | 必须引入 RT Pipeline |
| 平台限制 | NVIDIA GPU (OptiX) | NVIDIA GPU (VK_NV_ray_tracing_motion_blur) |

## 十三、当前实现状态

### 已完成
- [x] 设备扩展检测与启用
- [x] MotionInstance 数据模型
- [x] PrimitiveBase 类型标签系统
- [x] BLAS 顶点运动模糊（2 关键帧）
- [x] TLAS motion instance buffer（184 字节 STATIC 类型）
- [x] TLAS motion build flags
- [x] Ray Tracing Pipeline 基础设施
- [x] SPIR-V 后处理（OpTraceRayKHR → OpTraceRayMotionNV）
- [x] HLSL codegen：motion blur trace
- [x] 命令重排序：MotionInstanceBuildCommand
- [x] Builtin kernel：accel_process_vk_motion

### 限制与待改进
- [ ] Motion instance 当前仅写入 STATIC 类型（type=0），未利用 SRT/MATRIX motion 变体
- [ ] 仅支持 2 个关键帧（Vulkan NV 扩展限制）
- [ ] Query motion blur 操作仍降级为非 motion 版本
- [ ] Instance Motion SRT/Matrix 设备端读写未实现
- [ ] 所有 motion instance 在 TLAS 中退化为静态实例（关键帧变换数据未写入 transformT0/T1）


## 参考资料
+ [https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#traceray](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#traceray)
+ [https://microsoft.github.io/DirectX-Specs/d3d/Raytracing2.html](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing2.html)
+ [https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_ray_tracing_motion_blur.html](https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_ray_tracing_motion_blur.html)
+ [https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)