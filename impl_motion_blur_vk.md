# LuisaCompute：Motion Blur Vulkan 后端实现报告

## 第一章：test_motion_blur.cpp 测试代码分析

> 源文件：`./src/tests/integration/runtime/test_motion_blur.cpp`

### 1.1 测试概述

该测试展示了 LuisaCompute 中渲染 motion blur：

| 特性 | 说明 |
|------|------|
| Mesh 运动模糊 | 顶点动画，三角形网格的顶点在多个关键帧之间插值，产生几何形变模糊 |
| Curve 运动模糊 | Catmull-Rom 曲线的控制点在关键帧之间插值 |
| Motion Instance SRT 变换 | 实例级别的 Scale-Rotate-Translate 动画变换 |

渲染分辨率 512×512，采样 1024 spp，使用渐进式累积。

### 1.2 场景构建细节

#### 1.2.1 Curve（曲线）— 2 个关键帧

```
控制点数量: 50
曲线基函数: CurveBasis::CATMULL_ROM（每段 4 个控制点）
段数: 50 - 4 + 1 = 47
关键帧数: 2
```

- 控制点按螺旋线分布（`cos/sin` 生成 x/z，线性递增 y）
- 第 4 分量 `w` 为曲线半径，随参数正弦变化
- 两个关键帧之间，y 坐标偏移 0.1（`y + k * 0.1f`），模拟曲线整体上移
- 控制点 buffer 大小 = `50 * 2 = 100` 个 `float4`

创建时通过 `AccelOption.motion.keyframe_count = 2` 启用运动模糊：

```cpp
AccelOption curve_option;
curve_option.motion.keyframe_count = curve_keyframe_count; // 2
auto curve = device.create_curve(curve_basis, control_point_buffer, segment_buffer, curve_option);
```

#### 1.2.2 Mesh（三角形网格）— 3 个关键帧

```
顶点数/帧: 3（单个三角形）
关键帧数: 3
总顶点数: 9
```

- 三个关键帧中，底边两个顶点固定，顶部顶点 x 坐标从 -0.1 → 0.0 → 0.1 变化
- 顶部顶点 y 坐标从 0.5 → 0.55 → 0.5 变化
- 效果：三角形顶部左右摆动 + 轻微拉伸

```cpp
AccelOption mesh_option;
mesh_option.motion.keyframe_count = mesh_keyframe_count; // 3
mesh_option.motion.time_start = 0.f;
mesh_option.motion.time_end = 1.f;
auto mesh = device.create_mesh(vertex_buffer, triangle_buffer, mesh_option);
```

关键点：vertex buffer 包含所有关键帧的顶点数据（连续存储），硬件根据 `keyframe_count` 自动将 buffer 等分为 N 份。

#### 1.2.3 Motion Instance（运动实例）— SRT 模式，3 个关键帧

```
子对象: curve（上面创建的曲线）
运动模式: AccelMotionMode::SRT
关键帧数: 3
```

- 每个关键帧绕 Y 轴旋转 `i * 15°`（四元数表示）
- 平移 `(0, -0.5, 0)`
- 缩放和剪切保持默认

```cpp
AccelMotionOption motion_option;
motion_option.mode = AccelMotionMode::SRT;
motion_option.keyframe_count = 3u;
auto motion_instance = device.create_motion_instance(curve, motion_option);
```

#### 1.2.4 TLAS（顶层加速结构）组装

```cpp
auto accel = device.create_accel();
accel.emplace_back(mesh, translation(-.3f, 0.f, 0.f) * scaling(2.f));  // 实例 0: 静态变换的 mesh
accel.emplace_back(motion_instance);                                     // 实例 1: 运动实例包裹的 curve
```

构建顺序：`curve.build()` → `mesh.build()` → `motion_instance.build()` → `accel.build()`

### 1.3 Shader 逻辑

核心是 `raytracing_kernel`：

1. 生成随机数 `u = rand(frame_index, coord)`，其中 `u.xy()` 用于像素抖动，`u.z` 用于时间采样
2. `time = u.z * 1.f` — 在 `[0, 1]` 区间内随机采样快门时间
3. 调用 `accel.intersect_motion(ray, time, {.curve_bases = {curve_basis}})` — 带时间参数的光线追踪
4. 根据命中类型着色：
   - 三角形：重心坐标插值 RGB
   - 曲线：按曲线参数从黑到白渐变
5. 渐进式累积：`lerp(old, color, 1/(frame_index+1))`

### 1.4 测试覆盖的 API 调用链

```
device.create_curve()           → DeviceInterface::create_curve()
device.create_mesh()            → DeviceInterface::create_mesh()       (带 motion option)
device.create_motion_instance() → DeviceInterface::create_motion_instance()
device.create_accel()           → DeviceInterface::create_accel()

curve.build()                   → CurveBuildCommand
mesh.build()                    → MeshBuildCommand                     (带 motion keyframes)
motion_instance.build()         → MotionInstanceBuildCommand
accel.build()                   → AccelBuildCommand

accel.intersect_motion()        → CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR
```




# 第二章：CUDA 后端运动模糊的实现细节

## 2.1 概述

CUDA 后端的运动模糊实现建立在 **NVIDIA OptiX 7+** 光线追踪引擎之上。其核心思路是：将 LuisaCompute 运行时层的 `MotionInstance` 抽象映射为 OptiX 的 **Motion Transform Traversable**（运动变换可遍历节点），在 IAS（Instance Acceleration Structure）中以 traversable handle 的形式引用，从而让 OptiX 硬件在光线遍历时自动完成关键帧之间的插值。

### 涉及的关键源文件：

| 文件 | 职责 |
| :--- | :--- |
| `cuda_motion_instance.h/cpp` | `CUDAMotionInstance` 类：GPU 端 motion transform buffer 的分配、构建 |
| `cuda_primitive.h/cpp` | `CUDAPrimitiveBase` 基类（含 `MOTION_INSTANCE` tag）与 `CUDAPrimitive` GAS 构建 |
| `cuda_accel.h/cpp` | `CUDAAccel` IAS 构建，处理 motion instance 的 handle 更新与 rebuild 判定 |
| `cuda_shader_optix.cpp` | OptiX pipeline 编译，`usesMotionBlur` 标志与 traversable graph 配置 |
| `cuda_command_encoder.cpp` | `MotionInstanceBuildCommand` 的 visitor 分发 |
| `optix_api.h` | OptiX 类型镜像：`SRTData`、`SRTMotionTransform`、`MatrixMotionTransform`、`MotionOptions` |
| `cuda_device_resource.h` | 设备端内建函数：`lc_accel_trace_*_motion_blur`、`lc_accel_instance_motion_srt` |
| `cuda_shader_metadata.h` | `requires_motion_blur` 元数据标志 |

---

## 2.2 MotionInstanceTransformSRT 数据结构

### 2.2.1 运行时层定义

`MotionInstanceTransformSRT` 定义于 `include/luisa/runtime/rtx/motion_transform.h:13`：

```cpp
struct alignas(16) MotionInstanceTransformSRT {
    float pivot[3]       = {0.f, 0.f, 0.f};      // 旋转枢轴点
    float quaternion[4]  = {0.f, 0.f, 0.f, 1.f}; // 旋转四元数 (x, y, z, w)
    float scale[3]       = {1.f, 1.f, 1.f};       // 缩放因子
    float shear[3]       = {0.f, 0.f, 0.f};       // 剪切参数
    float translation[3] = {0.f, 0.f, 0.f};       // 平移向量
};
```
该结构体严格保证 sizeof == 64、alignof == 16，通过 static_assert 在编译期校验。它与 MotionInstanceTransformMatrix（即 float4x4）共享同一个 64 字节的 union 容器 MotionInstanceTransform，通过 as_srt() / as_matrix() 进行 reinterpret_cast 访问。


### 2.2.2 与 OptiX SRTData 的字段映射
OptiX 的 SRTData 定义于 optix_api.h:573：
```
struct SRTData {
    float sx, a, b, pvx, sy, c, pvy, sz, pvz, qx, qy, qz, qw, tx, ty, tz;
};
```
这是一个 紧凑的 16-float 布局，字段顺序与 MotionInstanceTransformSRT 不同。在 cuda_motion_instance.cpp:105-124 的 build 过程中，逐字段进行转换：
| LuisaCompute 字段      | → | OptiX SRTData 字段 |
|----------------------|---|-------------------|
| scale[0]             | → | sx                |
| shear[0]             | → | a                 |
| shear[1]             | → | b                 |
| pivot[0]             | → | pvx               |
| scale[1]             | → | sy                |
| shear[2]             | → | c                 |
| pivot[1]             | → | pvy               |
| scale[2]             | → | sz                |
| pivot[2]             | → | pvz               |
| quaternion[0..3]     | → | qx, qy, qz, qw    |
| translation[0..2]    | → | tx, ty, tz        |
OptiX 对 SRT 变换的语义定义为：对于一个点 $\mathbf{p}$，变换结果为：
$$T \cdot R \cdot S \cdot (\mathbf{p} - \text{pivot}) + \text{pivot}$$
其中 $S$ 包含缩放和剪切，$R$ 是四元数旋转，$T$ 是平移。OptiX 在硬件层面对相邻关键帧的 SRT 参数进行球面线性插值（四元数 slerp）和线性插值（其余参数 lerp），这比矩阵模式的逐元素线性插值在旋转方面更加物理正确。

### 2.2.3 设备端镜像结构
在 GPU kernel 中，LCMotionSRT（cuda_device_resource.h:1333）是 MotionInstanceTransformSRT 的设备端镜像：
```
struct alignas(16) LCMotionSRT {
    lc_array<lc_float, 3> m0;  // pivot
    lc_array<lc_float, 4> m1;  // quaternion
    lc_array<lc_float, 3> m2;  // scale
    lc_array<lc_float, 3> m3;  // shear
    lc_array<lc_float, 3> m4;  // translation
};
```
设备端的读写函数 lc_accel_instance_motion_srt / lc_accel_set_instance_motion_srt 负责在 LCSRTData（OptiX 布局）和 LCMotionSRT（LuisaCompute 布局）之间进行字段重排。

## 2.3 CUDAMotionInstance 的生命周期
### 2.3.1 创建（构造函数）
CUDAMotionInstance::CUDAMotionInstance（cuda_motion_instance.cpp:12）执行以下步骤：
1. 计算 buffer 大小：根据 motion mode 选择 OptiX 结构体类型：
   - MotionMode::MATRIX → TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM，buffer 大小 = header(32B) + float[12] × keyframe_count
   - MotionMode::SRT → TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM，buffer 大小 = header(32B) + SRTData(64B) × keyframe_count
2. 分配 GPU 内存：cuMemAlloc(&_motion_buffer, buffer_size)
3. 注册 traversable handle：调用 optix::api().convertPointerToTraversableHandle(optix_ctx, _motion_buffer, traversable_type, &_handle)，将 GPU 内存指针转换为 OptiX 可遍历句柄。
关键点：OptiX 的 motion transform 是一种 指针型 traversable——它不需要通过 accelBuild 构建 BVH，而是直接将 GPU 内存地址注册为 traversable。这意味着每次更新关键帧数据只需要一次 cuMemcpyHtoDAsync，无需重建加速结构。


### 2.3.2 构建（填充关键帧数据）
CUDAMotionInstance::build（cuda_motion_instance.cpp:50）在 MotionInstanceBuildCommand 被 dispatch 时调用：
1. 记录子节点：_child = reinterpret_cast<CUDAPrimitive *>(command->child())，子节点是一个 GAS（Mesh/Curve/ProceduralPrimitive）。
2. 通过 upload buffer 填充 motion transform 数据：使用 encoder.with_upload_buffer() 分配 host-pinned 内存，在其中构造 OptiX 要求的内存布局：

| 偏移量 | 大小 | 字段 | 说明 |
|--------|------|------|------|
| 0 | 8 bytes | `TraversableHandle child` | 子 GAS 的 traversable handle |
| 8 | 8 bytes | `MotionOptions` | 包含 `numKeys`, `flags`, `timeBegin`, `timeEnd` |
| 16 | 12 bytes | `unsigned int pad[3]` | 填充 |
| 28 | 4 bytes | `padding to 32` | 填充至 32 字节边界 |
| 32 | N×stride | `keyframe data[]` | `SRTData[N]` 或 `float[12][N]` |
这个布局对应 MotionTransform 结构体（cuda_motion_instance.cpp:67），其 static_assert(sizeof == 32) 确保 header 部分恰好 32 字节。
1. MotionOptions 配置：
   - numKeys：关键帧数量
   - flags：MOTION_FLAG_START_VANISH / MOTION_FLAG_END_VANISH（控制时间范围外物体是否消失）
   - timeBegin / timeEnd：运动时间区间
2. 异步上传：cuMemcpyHtoDAsync(buffer, view->address(), size, stream)


### 2.3.3 销毁
析构函数简单地释放 GPU 内存：cuMemFree(_motion_buffer)。
## 2.4 OptiX 遍历结构的构建
### 2.4.1 遍历图拓扑
启用运动模糊后，OptiX 的 traversable graph 变为三层结构：
```
IAS (Instance Acceleration Structure)
├── optix::Instance[0] → GAS (静态几何)
├── optix::Instance[1] → MotionTransform traversable → GAS (运动几何)
└── optix::Instance[2] → MotionTransform traversable → GAS (运动几何)
```

对比无运动模糊时的两层结构（IAS → GAS），运动模糊在 instance 和 GAS 之间插入了一个 MotionTransform 节点。这就是为什么 cuda_shader_optix.cpp:291 中 max_traversal_depth 从 2 变为 3：
auto max_traversal_depth = metadata.requires_motion_blur ? 3u : 2u;

### 2.4.2 IAS 构建中的 Motion Instance 处理
CUDAAccel::build（cuda_accel.cpp:152）在处理 instance 修改时，对 motion instance 有特殊逻辑：
1. Primitive handle 解析（cuda_accel.cpp:219-225）：当检测到 primitive 的 tag 为 MOTION_INSTANCE 时，不直接使用 primitive 本身的 handle，而是通过 instance->child() 获取其子 GAS 的信息来设置 SBT 标志（procedural/curve 类型标记）。
2. Motion instance child 变更检测（cuda_accel.cpp:250-257）：维护 _motion_instance_to_primitive 映射表，检测 motion instance 的子节点是否发生变化。如果变化，强制触发 IAS rebuild。
3. Rebuild 判定（cuda_accel.cpp:261-267）：以下任一条件为真时触发 rebuild：
   - any_motion_instance_child_changed
   - 用户请求 FORCE_BUILD
   - 不允许 update
   - handle 变更等
4. Instance buffer 中的 handle：对于 motion instance，写入 instance buffer 的 traversableHandle 是 CUDAMotionInstance::handle()（即 motion transform traversable），而非子 GAS 的 handle。OptiX 在遍历时会自动跟随 motion transform → child GAS 的链接。


### 2.4.3 GAS 层面的运动模糊（顶点动画）
除了 instance-level 的 motion transform，LuisaCompute 还支持 GAS-level 的运动模糊（顶点关键帧）。这通过 CUDAPrimitive 的 motion 选项实现：
- make_optix_build_options（cuda_primitive.h:68）将 AccelOption::motion 中的关键帧参数写入 optix::AccelBuildOptions::motionOptions
- CUDAPrimitive::_motion_buffer_pointers（cuda_primitive.cpp:134）将连续的顶点 buffer 按关键帧数量等分为多个指针，供 OptiX 的 vertexBuffers 数组使用
这两种运动模糊可以叠加：一个 mesh 可以同时具有顶点动画（GAS motion）和实例变换动画（motion transform）。


## 2.5 OptiX Pipeline 的运动模糊配置
CUDAShaderOptiX 构造函数（cuda_shader_optix.cpp:49）中，运动模糊影响 pipeline 编译的三个关键参数：
```
pipeline_compile_options.usesMotionBlur = metadata.requires_motion_blur;
pipeline_compile_options.traversableGraphFlags =
    metadata.requires_motion_blur ?
        optix::TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY :                    // 允许任意遍历图
        optix::TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING; // 仅两层
```
- usesMotionBlur = true 告知 OptiX 编译器生成支持时间参数的遍历代码
- TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY 是必须的，因为 motion transform 节点使遍历图不再是简单的单层 instancing
此外，内建 intersection shader 模块（用于曲线等非三角形几何）也需要声明 motion blur 支持：
```
optix::BuiltinISOptions options{
    .builtinISModuleType = type,
    .usesMotionBlur = metadata.requires_motion_blur,  // ← 关键
    ...
};
```
requires_motion_blur 标志来源于 AST 分析阶段：编译器检测 kernel 是否调用了 RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR 或 RAY_TRACING_TRACE_ANY_MOTION_BLUR 等 CallOp，并将结果记录在 CUDAShaderMetadata::requires_motion_blur 中。


## 2.6 设备端光线追踪调用
运动模糊的光线追踪在设备端通过 lc_accel_trace_closest_motion_blur / lc_accel_trace_any_motion_blur（cuda_device_resource.h:1855-1866）实现。它们与非运动模糊版本的唯一区别是 传入了 time 参数：
```
auto lc_accel_trace_closest_motion_blur(LCAccel accel, LCRay ray, lc_float time, lc_uint mask) {
    constexpr auto flags = LC_RAY_FLAG_DISABLE_ANYHIT | LC_RAY_FLAG_DISABLE_CLOSESTHIT;
    return lc_accel_trace_closest_impl<flags>(accel, ray, time, mask);
}
```
time 参数最终传递给 lc_ray_traverse 模板函数（cuda_device_resource.h:1765），该函数通过内联 PTX 汇编调用 _optix_hitobject_traverse，将 time 作为浮点参数传入 OptiX 硬件遍历单元。OptiX 根据此时间值在 motion transform 的关键帧之间进行插值，并对 motion GAS 选择对应时间的顶点数据。
对于 ray query 模式，lc_accel_query_all_motion_blur / lc_accel_query_any_motion_blur 同样将 time 存入 LCRayQuery 结构体，在后续的 lc_ray_query_trace 中传递给遍历函数。


## 2.7 Codegen 层面的支持
AST codegen（cuda_codegen_ast.cpp:1061-1120）和 XIR codegen（cuda_codegen_xir.cpp）将 DSL 层的运动模糊操作映射为设备端函数名：
CallOp	生成的设备函数
```
RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR	lc_accel_trace_closest_motion_blur
RAY_TRACING_TRACE_ANY_MOTION_BLUR	    lc_accel_trace_any_motion_blur
RAY_TRACING_QUERY_ALL_MOTION_BLUR	    lc_accel_query_all_motion_blur
RAY_TRACING_QUERY_ANY_MOTION_BLUR	    lc_accel_query_any_motion_blur
RAY_TRACING_INSTANCE_MOTION_SRT	        lc_accel_instance_motion_srt
RAY_TRACING_SET_INSTANCE_MOTION_SRT	    lc_accel_set_instance_motion_srt
RAY_TRACING_INSTANCE_MOTION_MATRIX	    lc_accel_instance_motion_matrix
RAY_TRACING_SET_INSTANCE_MOTION_MATRIX	lc_accel_set_instance_motion_matrix
```
MotionInstanceTransformSRT 类型在 codegen 中被识别为特殊类型（_motion_srt_type），生成对应的 LCMotionSRT 设备端类型名。


## 2.8 小结
CUDA 后端的运动模糊实现可以概括为以下数据流：
```
用户设置 MotionInstanceTransformSRT 关键帧
        │
        ▼
MotionInstanceBuildCommand (运行时命令)
        │
        ▼
CUDAMotionInstance::build()
  ├── 字段重排: MotionInstanceTransformSRT → optix::SRTData
  ├── 填充 MotionTransform header (child handle, MotionOptions)
  └── cuMemcpyHtoDAsync → GPU motion buffer
        │
        ▼
CUDAAccel::build()
  ├── Instance buffer 中写入 motion transform 的 traversable handle
  └── optix::accelBuild() 构建 IAS
        │
        ▼
OptiX Pipeline (usesMotionBlur=true, ALLOW_ANY graph)
        │
        ▼
设备端 lc_accel_trace_*_motion_blur(accel, ray, time, mask)
  └── _optix_hitobject_traverse(..., time, ...)
        │
        ▼
OptiX 硬件自动插值 SRT 关键帧，遍历变换后的 GAS
```

与 Vulkan 后端相比，CUDA/OptiX 后端的运动模糊具有以下特点：
- Motion transform 是 指针型 traversable，无需 BVH 构建，更新成本仅为一次 memcpy
- SRT 插值由 OptiX RT Core 硬件完成，四元数旋转使用 slerp
- 遍历图深度从 2 增加到 3，需要相应调整 pipeline stack size
- time 参数通过 PTX 内联汇编直接传递给硬件遍历单元



# 第三章：Vulkan 后端 Motion Blur 实现差距分析
## 3.1 概述
基于当前工作区的 git 改动（含已修改文件和新增文件），Vulkan 后端的 motion blur 实现处于早期脚手架阶段：设备层扩展检测与数据模型已就绪，但加速结构构建和着色器两个核心环节均未完成。本章逐层分析已完成的工作和尚缺的功能，并与第二章的 CUDA/OptiX 实现进行对比。

## 3.2 已完成的工作
### 3.2.1 设备层扩展检测与启用（已完成）
device.cpp 新增了对 VK_NV_ray_tracing_motion_blur 和 VK_KHR_ray_tracing_pipeline 两个扩展的检测与启用逻辑：
```
// device.cpp:578-586
if (supported_ext.find(VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME) != supported_ext.end() &&
    supported_ext.find(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) != supported_ext.end()) {
    _enable_device_exts.emplace_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    _enable_device_exts.emplace_back(VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME);
    enable_motion_blur = true;
    motion_blur_enabled = true;
}
```
同时在设备创建的 feature chain 中链入了
`VkPhysicalDeviceRayTracingMotionBlurFeaturesNV（rayTracingMotionBlur = VK_TRUE）` 和 `VkPhysicalDeviceRayTracingPipelineFeaturesKHR（rayTracingPipeline = VK_TRUE）`。device.h 新增了 motion_blur_enabled 位域和 enable_motion_blur() 访问器。
对比 CUDA：CUDA 后端无需显式启用扩展，OptiX 原生支持 motion blur。Vulkan 后端需要依赖 NVIDIA 专有扩展 VK_NV_ray_tracing_motion_blur，这意味着该功能仅在 NVIDIA GPU 上可用。

### 3.2.2 PrimitiveBase 类型标签系统（已完成）
新增 primitive_base.h，引入 PrimitiveBase 基类，为 TLAS 提供区分 Blas（静态几何）和 MotionInstance（运动实例）的能力：
```
// primitive_base.h
class PrimitiveBase : public Resource {
public:
    enum class PrimTag { BLAS, MOTION_INSTANCE };
    [[nodiscard]] bool is_motion_instance() const noexcept;
};
```
Blas 类的继承关系从 Resource 改为 PrimitiveBase（blas.h），构造时传入 PrimTag::BLAS。
对比 CUDA：CUDA 后端的 CUDAPrimitiveBase 有类似的 tag 系统（MESH、PROCEDURAL、CURVE、MOTION_INSTANCE），但粒度更细。
3.2.3 MotionInstance 数据模型（已完成）
新增 motion_instance.h/cpp，MotionInstance 类存储：
- AccelMotionOption：运动模式（SRT/Matrix）、关键帧数、时间范围、vanish 标志
- Blas *_child：子 BLAS 指针
- vector<MotionInstanceTransform> _keyframes：关键帧变换数据（每帧 64 字节）
构造函数校验设备是否支持 motion blur 以及关键帧数 ≥ 2。
对比 CUDA：CUDAMotionInstance 除了存储关键帧外，还在构造时分配 GPU 内存并注册 OptiX traversable handle。Vulkan 的 MotionInstance 目前仅在 CPU 端存储数据，无 GPU 资源。

### 3.2.4 命令分发（已完成）
stream.cpp 处理 MotionInstanceBuildCommand：
```
// stream.cpp:1182-1189 (第一遍扫描)
auto c = static_cast<MotionInstanceBuildCommand const *>(command.get());
auto mi = reinterpret_cast<MotionInstance *>(c->handle());
mi->set_child(reinterpret_cast<Blas *>(c->child()));
mi->set_keyframes(const_cast<MotionInstanceBuildCommand *>(c)->steal_keyframes());
```
在预处理和执行阶段均为空操作（注释标注 "Already handled" / "no GPU work needed"）。
command_reorder_visitor.h 中将 MotionInstanceBuildCommand 注册为对 motion instance handle 的写操作，确保在 AccelBuildCommand 之前完成排序。
对比 CUDA：CUDA 后端在 build() 阶段执行实际的 GPU 工作——填充 OptiX MotionTransform header + 关键帧数据，然后 cuMemcpyHtoDAsync 上传到 GPU。

### 3.2.5 TLAS 中的 MotionInstance 解析（部分完成）
tlas.cpp 新增 resolve_to_blas() 辅助函数，当 primitive handle 是 MotionInstance 时，提取其子 Blas*：
```
// tlas.cpp:21-28
static Blas *resolve_to_blas(uint64_t primitive_handle) {
    auto prim = reinterpret_cast<PrimitiveBase *>(primitive_handle);
    if (prim->is_motion_instance()) {
        auto mi = static_cast<MotionInstance *>(prim);
        return mi->child();
    }
    return static_cast<Blas *>(prim);
}
```
修改后的 pre_build() 使用 resolved_meshes 数组统一处理 primitive 解析，避免直接 reinterpret_cast<Blas*>(i.primitive)。
问题：当前实现将 MotionInstance 退化为其子 BLAS 的静态引用，关键帧数据被完全忽略。

### 3.2.6 HLSL Codegen 的 Motion Blur Stub（部分完成）
function_codegen.cpp 新增了 4 个 motion blur CallOp 的处理，但采用丢弃 time 参数的降级策略：
```
// function_codegen.cpp:768-778
case CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
    str << "_TraceClosest("sv;
    args[0]->accept(vis);  // accel
    str << ',';
    args[1]->accept(vis);  // ray
    str << ',';
    args[3]->accept(vis);  // mask (跳过 args[2] 即 time)
    str << ')';
    return;
```
RAY_TRACING_TRACE_ANY_MOTION_BLUR、RAY_TRACING_QUERY_ALL_MOTION_BLUR、RAY_TRACING_QUERY_ANY_MOTION_BLUR 同理。
效果：shader 可以编译通过，但运动模糊不生效——所有物体按静态位置渲染。

## 3.3 尚未实现的功能
### 3.3.1 TLAS 构建：Motion Instance Buffer 布局（核心缺失）
现状：TLAS 的 instance buffer 使用标准 VkAccelerationStructureInstanceKHR（64 字节/实例），所有实例均为静态。

需要：当存在 motion instance 时，整个 instance buffer 必须切换为 VkAccelerationStructureMotionInstanceNV（152 字节/实例）。这是 VK_NV_ray_tracing_motion_blur 扩展的硬性要求——一旦 TLAS 启用 motion，所有实例（包括静态实例）都必须使用 152 字节的 motion instance 格式。
VkAccelerationStructureMotionInstanceNV 的布局：
| 偏移量 | 大小 | 字段 | 说明 |
|--------|------|------|------|
| 0 | 4 bytes | type | STATIC_NV(0) / MATRIX_MOTION_NV(1) / SRT_MOTION_NV(2) |
| 4 | 4 bytes | flags | 保留，必须为 0 |
| 8 | 144 bytes | data (union) | 见下方 |
data union 的三种变体：
- Static（type=0）：内嵌标准 VkAccelerationStructureInstanceKHR（64 字节），剩余 80 字节填充
- Matrix Motion（type=1）：transformT0（3×4 float）+ transformT1（3×4 float）+ instance 字段
- SRT Motion（type=2）：transformT0（VkSRTDataNV，64 字节）+ transformT1（VkSRTDataNV，64 字节）+ instance 字段

关键限制：Vulkan NV 扩展仅支持恰好 2 个关键帧（transformT0 和 transformT1），而 OptiX 支持 N 个关键帧。LuisaCompute 运行时 API 的 AccelMotionOption.keyframe_count 可以 > 2，Vulkan 后端需要：
- 限制为 2 个关键帧（取首尾），或
- 对 > 2 关键帧的情况报错/降级
需要修改的代码：
  1. tlas.cpp:45 — instance buffer 大小计算需根据是否有 motion instance 选择 64 或 152 字节步长
  2. tlas.cpp:222-230 — VkAccelerationStructureGeometryInstancesDataKHR 的数据指向 motion instance buffer
  3. tlas.cpp:232-240 — build flags 需添加 VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV
  4. tlas.cpp:266-276 — create info 需添加 VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV
  5. accel_process_vk（内建 compute shader）— 需要 motion 变体，写入 152 字节的 VkAccelerationStructureMotionInstanceNV，包括 type 判别、SRT/Matrix 关键帧数据
  6. tlas.cpp 的 modification 循环 — 对 MotionInstance 类型的 primitive，需要提取关键帧数据并传入 instance buffer
对比 CUDA：OptiX 的 motion transform 是独立的 traversable 节点（指针型），不改变 IAS instance 的大小。Vulkan 的方案将 motion 数据内嵌到 instance 结构体中，导致所有实例的步长统一增大到 152 字节。

### 3.3.2 SRT 数据字段重排（未实现）
VkSRTDataNV 的字段布局与 MotionInstanceTransformSRT 不同：
```
MotionInstanceTransformSRT 字段	→	VkSRTDataNV 字段
scale[0]	→	sx
shear[0]	→	a
shear[1]	→	b
pivot[0]	→	pvx
scale[1]	→	sy
shear[2]	→	c
pivot[1]	→	pvy
scale[2]	→	sz
pivot[2]	→	pvz
quaternion[0..3]	→	qx, qy, qz, qw
translation[0..2]	→	tx, ty, tz
```
这与 CUDA 后端 cuda_motion_instance.cpp:105-124 中的 OptiX SRTData 转换完全一致（VkSRTDataNV 与 OptiX SRTData 布局相同，均为 16-float 紧凑排列）。Vulkan 后端需要实现相同的字段重排逻辑。

### 3.3.3 着色器层：Motion Blur 光线追踪（核心缺失）
现状：HLSL codegen 将 motion blur trace/query 操作降级为非 motion 版本，丢弃 time 参数。
根本问题：HLSL 的 inline ray tracing API（RayQuery<>::TraceRayInline()）不支持 time 参数。这是 DXR 1.1 inline ray tracing 的固有限制。VK_NV_ray_tracing_motion_blur 扩展的 time 参数仅在以下两种方式中可用：
1. Pipeline-based ray tracing（TraceRay() / OpTraceRayMotionNV）— 需要完整的 ray tracing pipeline（RayGen + ClosestHit + Miss shader），而非当前使用的 compute shader + inline ray query
2. SPIR-V 扩展 SPV_NV_ray_tracing_motion_blur — 提供 OpTraceRayMotionNV 指令，但这是 pipeline trace 指令，不适用于 inline ray query
可能的解决方案：

| 方案 | 描述 | 复杂度 | 可行性 |
|------|------|--------|--------|
| A. Pipeline-based RT | 将 motion blur shader 从 compute + inline ray query 切换为完整的 ray tracing pipeline | 极高 | 需要重构整个 VK 后端的 RT 架构 |
| B. SPIR-V 后处理 | DXC 编译后，对 SPIR-V 进行 patch，将 OpRayQueryInitializeKHR 替换为带 time 参数的变体 | 高 | SPV_NV_ray_tracing_motion_blur 未定义 inline ray query 的 motion 变体 |
| C. 等待 Vulkan 扩展 | 等待 Vulkan 标准或 NV 扩展支持 inline ray query 的 time 参数 | — | 取决于 Khronos/NVIDIA 路线图 |
| D. DXC HLSL 扩展 | 使用 DXC 的 SPIR-V 扩展机制（-fspv-extension=SPV_NV_ray_tracing_motion_blur）+ HLSL 内联 SPIR-V | 中 | 需要验证 DXC 是否支持 |
当前 DXC 编译配置（shader_compiler.cpp:149-154）：
```
args.emplace_back(L"-spirv");
args.emplace_back(L"/DSPV");
args.emplace_back(L"-fspv-target-env=vulkan1.1");
```
未传入任何 SPIR-V 扩展标志。即使 HLSL 代码中使用了 motion blur 内建函数，DXC 也不会生成对应的 SPIR-V 指令。

### 3.3.4 BLAS 层面的顶点运动模糊（未实现）
现状：blas.cpp 和 blas.h 中无任何 motion 相关代码。BLAS 构建仅处理单帧顶点数据。
需要：LuisaCompute 运行时支持 mesh/curve/procedural primitive 的多关键帧顶点数据（通过 AccelOption.motion.keyframe_count 配置）。CUDA 后端通过 CUDAPrimitive::_motion_buffer_pointers() 将连续的顶点 buffer 按关键帧数等分，供 OptiX 的 vertexBuffers 数组使用。
Vulkan 的 VK_NV_ray_tracing_motion_blur 扩展同样支持 BLAS 级别的 motion：在 VkAccelerationStructureBuildGeometryInfoKHR 中设置 VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV，并通过 VkAccelerationStructureGeometryMotionTrianglesDataNV 提供多帧顶点数据。
需要修改的代码：
- blas.cpp 的 mesh 构建路径：检测 AccelOption.motion，设置 motion build flag，提供多帧顶点指针
- blas.h：存储 motion 选项

### 3.3.5 Instance Motion SRT/Matrix 的设备端读写（未实现）
现状：HLSL codegen（function_codegen.cpp）未处理以下 4 个 CallOp：
- RAY_TRACING_INSTANCE_MOTION_SRT — 读取 motion instance 的 SRT 关键帧
- RAY_TRACING_SET_INSTANCE_MOTION_SRT — 写入 motion instance 的 SRT 关键帧
- RAY_TRACING_INSTANCE_MOTION_MATRIX — 读取 motion instance 的 Matrix 关键帧
- RAY_TRACING_SET_INSTANCE_MOTION_MATRIX — 写入 motion instance 的 Matrix 关键帧
对比 CUDA：CUDA 后端在 cuda_device_resource.h 中实现了 lc_accel_instance_motion_srt / lc_accel_set_instance_motion_srt 等函数，通过解引用 OptiX traversable handle 的指针直接读写 GPU 内存中的关键帧数据。
Vulkan 后端需要在 accel_header.bytes 中新增对应的 HLSL 函数，操作 motion instance buffer 中的 SRT/Matrix 数据。由于 Vulkan 的 motion instance 数据内嵌在 TLAS instance buffer 中（而非独立的 traversable），读写逻辑需要按 152 字节步长索引并偏移到 data union 的对应位置。

### 3.3.6 Shader Metadata 的 Motion Blur 标志（未实现）
现状：Vulkan 后端的 shader metadata 中没有 requires_motion_blur 标志。
对比 CUDA：CUDAShaderMetadata::requires_motion_blur 由 AST 分析阶段设置，用于控制 OptiX pipeline 的 usesMotionBlur 和 traversableGraphFlags。
Vulkan 后端如果采用 pipeline-based RT 方案，同样需要此标志来决定是否启用 motion blur 相关的 pipeline 配置。
3.4 差距总结
| 层次 | 功能 | 状态 | 阻塞因素 |
|------|------|------|----------|
| 设备扩展检测 | VK_NV_ray_tracing_motion_blur 启用 | ✅ 已完成 | — |
| 设备 feature chain | Motion blur + RT pipeline features | ✅ 已完成 | — |
| MotionInstance 数据模型 | CPU 端关键帧存储 | ✅ 已完成 | — |
| PrimitiveBase 类型标签 | BLAS / MOTION_INSTANCE 区分 | ✅ 已完成 | — |
| 命令分发 | MotionInstanceBuildCommand 处理 | ✅ 已完成 | — |
| 命令重排序 | MotionInstanceBuildCommand 排序 | ✅ 已完成 | — |
| TLAS primitive 解析 | resolve_to_blas() | ✅ 已完成 | — |
| HLSL codegen stub | Motion blur ops 降级为非 motion 版本 | ⚠️ 临时方案 | — |
| TLAS motion instance buffer | 152 字节 VkAccelerationStructureMotionInstanceNV | ❌ 未实现 | 需重写 instance buffer 布局 |
| TLAS motion build flags | MOTION_BIT_NV on build + create | ❌ 未实现 | 依赖 instance buffer 改造 |
| SRT 字段重排 | MotionInstanceTransformSRT → VkSRTDataNV | ❌ 未实现 | 依赖 instance buffer 改造 |
| accel_process_vk shader | Motion instance 写入 compute shader | ❌ 未实现 | 需要 motion 变体 |
| 着色器 motion blur trace | time 参数传递给硬件 | ❌ 未实现 | HLSL inline ray query 不支持 time |
| DXC SPIR-V 扩展 | SPV_NV_ray_tracing_motion_blur | ❌ 未配置 | 依赖着色器方案选择 |
| BLAS 顶点运动模糊 | 多关键帧顶点 buffer | ❌ 未实现 | 需改造 BLAS 构建 |
| Instance motion SRT/Matrix 读写 | 设备端关键帧访问 | ❌ 未实现 | 需新增 HLSL 函数 |
| Shader metadata | requires_motion_blur 标志 | ❌ 未实现 | 依赖着色器方案选择 |

## 3.5 核心技术障碍
当前 Vulkan 后端实现 motion blur 的最大技术障碍在于着色器层：
1. HLSL RayQuery<>::TraceRayInline() 不接受 time 参数。这是 DXR 1.1 inline ray tracing 的设计限制，不是 LuisaCompute 的问题。
2. VK_NV_ray_tracing_motion_blur 的 time 参数仅在 pipeline-based TraceRay 中可用（对应 SPIR-V 的 OpTraceRayMotionNV）。
3. LuisaCompute 的 Vulkan 后端当前使用 compute shader + inline ray query 架构，与 pipeline-based RT 架构不兼容。
这意味着要在 Vulkan 后端实现真正的 motion blur 光线追踪，可能需要：
- 引入 ray tracing pipeline 支持（RayGen/ClosestHit/Miss/AnyHit shader stages），或
- 等待 Vulkan 扩展支持 inline ray query 的 motion 变体，或
- 探索 SPIR-V 后处理方案
相比之下，TLAS 构建层面的改造（motion instance buffer、build flags、SRT 重排）虽然工作量不小，但技术路径清晰，不存在根本性障碍。



# 第四章：Vulkan 后端 Motion Blur 完整实现计划
## 4.1 架构决策：为什么必须引入 Ray Tracing Pipeline
报告第三章 §3.5 指出的核心技术障碍经过验证确认：
- SPV_NV_ray_tracing_motion_blur 仅定义了 OpTraceMotionNV / OpTraceRayMotionNV 两条指令，它们只允许在 RayGenerationKHR、ClosestHitKHR、MissKHR 执行模型中使用
- Vulkan/SPIR-V 生态中不存在 inline ray query 的 motion 变体（没有 OpRayQueryInitializeMotion 之类的指令）
- 当前 VK 后端使用 compute shader + RayQuery<>::TraceRayInline() 架构，TraceRayInline 不接受 time 参数
因此，对于使用了 motion blur trace 的 shader，必须切换到 ray tracing pipeline 架构（RayGen shader 通过 vkCmdTraceRaysKHR 发射光线）。不使用 motion blur 的 shader 可以保持现有的 compute + inline ray query 路径不变。
这是整个计划中工作量最大、风险最高的部分。
1. 分阶段实施路线
- 阶段 A: TLAS Motion Instance 构建（主机端，不涉及着色器）
- 阶段 B: BLAS 顶点运动模糊（主机端，不涉及着色器）
- 阶段 C: Ray Tracing Pipeline 基础设施（着色器架构变更）
- 阶段 D: Motion Blur Shader Codegen（HLSL/SPIR-V 代码生成）
- 阶段 E: Instance Motion SRT/Matrix 读写（设备端数据访问）
- 阶段 F: 集成测试与验证
---
阶段 A：TLAS Motion Instance 构建
目标：让 TLAS 正确构建包含 VkAccelerationStructureMotionInstanceNV 的加速结构。

A1. Tlas 类增加 motion 感知
文件：tlas.h, tlas.cpp
- 新增 bool _has_motion_instances 成员，在 pre_build() 中扫描 modifications 判定是否存在 MotionInstance
- 当 _has_motion_instances == true 时：
  - instance buffer 步长从 64 字节改为 152 字节（sizeof(VkAccelerationStructureMotionInstanceNV)）
  - VkAccelerationStructureBuildGeometryInfoKHR::flags 添加 VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV
  - VkAccelerationStructureCreateInfoKHR::createFlags 添加 VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV

A2. 新建 motion 版 accel_process compute shader
文件：新增 src/backends/common/hlsl/builtin/accel_process_vk_motion
- 输出 buffer 类型从 RWStructuredBuffer<MeshInst>（64B）改为 RWByteAddressBuffer（按 152B 步长手动寻址）
- 对每个 instance：
  - 静态实例：写入 type=0（STATIC），内嵌标准 VkAccelerationStructureInstanceKHR
  - SRT motion 实例：写入 type=2（SRT_MOTION），填充 transformT0 + transformT1（各 64 字节 VkSRTDataNV）
  - Matrix motion 实例：写入 type=1（MATRIX_MOTION），填充 transformT0 + transformT1（各 48 字节 3×4 矩阵）
- 输入 buffer 需要扩展，增加 motion type 标志和关键帧数据字段

A3. SRT 字段重排

文件：tlas.cpp（或独立的 motion_transform_convert.h）

实现 MotionInstanceTransformSRT → VkSRTDataNV 的转换函数，字段映射与 CUDA 后端 cuda_motion_instance.cpp:105-124 完全一致：
VkSRTDataNV to_vk_srt(const MotionInstanceTransformSRT &srt) {
    return {
        .sx = srt.scale[0], .a = srt.shear[0], .b = srt.shear[1], .pvx = srt.pivot[0],
        .sy = srt.scale[1], .c = srt.shear[2], .pvy = srt.pivot[1],
        .sz = srt.scale[2], .pvz = srt.pivot[2],
        .qx = srt.quaternion[0], .qy = srt.quaternion[1],
        .qz = srt.quaternion[2], .qw = srt.quaternion[3],
        .tx = srt.translation[0], .ty = srt.translation[1], .tz = srt.translation[2]
    };
}

A4. 关键帧数限制处理
VK_NV_ray_tracing_motion_blur 仅支持恰好 2 个关键帧（transformT0, transformT1）。当 AccelMotionOption.keyframe_count > 2 时：
- 取首尾两个关键帧（index 0 和 index N-1）
- 输出 warning 日志

A5. Modification 循环改造
文件：tlas.cpp 的 pre_build()
- 在 resolved_meshes 之外，新增 resolved_motion_instances 数组，记录每个 modification 对应的 MotionInstance*（如果有的话）
- 对 MotionInstance 类型的 primitive：提取关键帧数据、motion mode、time range，传入 staging buffer
- 扩展 TlasInputInst 结构体（或新建 TlasMotionInputInst），携带 motion type + 关键帧数据

A6. 注册 motion 版 builtin kernel
文件：builtin_kernel.cpp, device.h
- 新增 set_accel_motion_kernel LazyLoadShader，加载 accel_process_vk_motion
- tlas.cpp 根据 _has_motion_instances 选择使用哪个 kernel
---
阶段 B：BLAS 顶点运动模糊
目标：支持三角形 mesh 的多关键帧顶点数据。

B1. Blas 类存储 motion 选项
文件：blas.h, blas.cpp
- 在 mesh 构建路径中检测 AccelOption.motion.is_enabled()
- 存储 keyframe_count、time_start、time_end、vanish 标志

B2. BLAS 构建添加 motion 支持
文件：blas.cpp
- 当 motion 启用时：
  - VkAccelerationStructureBuildGeometryInfoKHR::flags 添加 VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV
  - VkAccelerationStructureCreateInfoKHR::createFlags 添加 VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV
  - 使用 VkAccelerationStructureGeometryMotionTrianglesDataNV 作为 pNext，提供第二帧顶点数据指针
  - 将连续的顶点 buffer 按关键帧数等分（参考 CUDA 的 _motion_buffer_pointers()）
注意：VkAccelerationStructureGeometryMotionTrianglesDataNV 同样只支持 2 个关键帧（vertexData 是基础帧，pNext 中的 motion data 是第二帧）。
---
阶段 C：Ray Tracing Pipeline 基础设施
目标：为 motion blur shader 引入 ray tracing pipeline 执行路径。
这是最大的架构变更。核心思路：当 shader metadata 标记 requires_motion_blur 时，不走 compute pipeline + inline ray query，而是走 ray tracing pipeline + OpTraceRayMotionNV。

C1. Shader Metadata 扩展
文件：VK 后端的 shader metadata（可能在 compute_shader.h 或新建 shader_metadata.h）
- 新增 bool requires_motion_blur 标志
- 在 shader 编译阶段，扫描 AST/XIR 中是否使用了 RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR 等 CallOp，设置此标志

C2. Ray Tracing Pipeline 创建
文件：新增 rt_shader.h, rt_shader.cpp（或扩展现有 compute_shader）
当 requires_motion_blur == true 时：
1. 将 shader 编译为 RayGen shader（cs_6_x → lib_6_x 或直接生成 SPIR-V raygen 入口）
2. 创建 VkRayTracingPipelineCreateInfoKHR：
   - RayGen shader group（用户的 kernel 逻辑）
   - Miss shader group（返回 miss 结果）
   - ClosestHit shader group（返回 hit 结果，包含三角形重心坐标等）
3. 创建 Shader Binding Table (SBT) buffer
4. 通过 vkCmdTraceRaysKHR 发射光线（替代 vkCmdDispatch）

C3. DXC 编译配置变更
文件：shader_compiler.cpp
- 当目标是 motion blur RT shader 时：
  - 添加 -fspv-extension=SPV_NV_ray_tracing_motion_blur
  - 添加 -fspv-extension=SPV_KHR_ray_tracing
  - shader model 改为 lib_6_5 或更高
  - target 改为 vulkan1.2（ray tracing pipeline 需要）

C4. 简化方案：仅 RayGen + Inline Ray Query with Motion
实际上有一个更轻量的方案值得优先尝试：
VK_NV_ray_tracing_motion_blur 的 Vulkan spec 中提到，rayTracingMotionBlur feature 启用后，ray query 在 motion TLAS 上的行为是：以 time=0 进行遍历。也就是说，即使用 inline ray query 对 motion TLAS 做查询，硬件也会正确处理（只是 time 固定为 0）。
但如果我们需要自定义 time 参数，就必须用 OpTraceRayMotionNV。
折中方案：使用 VkPipeline 的 ray tracing pipeline，但 shader 逻辑尽量简单——RayGen shader 中包含用户的完整 kernel 逻辑，ClosestHit/Miss shader 只做最小化的 payload 写入。这样可以最大限度复用现有的 HLSL codegen 基础设施。
---
阶段 D：Motion Blur Shader Codegen
目标：生成正确的 HLSL/SPIR-V 代码，将 time 参数传递给硬件。

D1. HLSL Codegen 修改
文件：function_codegen.cpp
将当前的 stub 实现替换为真正的 motion blur 调用：
// 替代当前的 _TraceClosest(accel, ray, mask)
// 生成使用 TraceRay + time 参数的代码
_TraceClosestMotion(accel, ray, time, mask)
具体生成的 HLSL 取决于阶段 C 选择的方案：
- 如果走 ray tracing pipeline：生成 TraceRay() 调用 + payload 结构体
- 如果走 SPIR-V inline asm：生成内联 SPIR-V 汇编

D2. raytracing_header.bytes 扩展
文件：src/backends/common/hlsl/builtin/raytracing_header.bytes
新增 motion 版本的 trace 函数：
// Pipeline-based motion blur trace
_Hit1 _TraceClosestMotion(RaytracingAccelerationStructure accel, T rayDesc, float time, uint mask) {
    // 使用 TraceRay with time parameter (via SPV_NV_ray_tracing_motion_blur)
    RayDesc ray = _toRay(rayDesc);
    _Hit1 payload = (_Hit1)0;
    TraceMotionRay(accel, RAY_FLAG_FORCE_OPAQUE, mask, 0, 0, 0, ray, time, payload);
    return payload;
}

D3. Query 变体
同理为 _QueryAllMotion / _QueryAnyMotion 生成代码。由于 ray query 不支持 time，这些需要通过 pipeline-based trace 实现，或者降级为 time=0 并输出 warning。

---
阶段 E：Instance Motion SRT/Matrix 设备端读写
目标：支持在 shader 中读写 motion instance 的关键帧数据。

E1. HLSL Codegen 处理 4 个 CallOp
文件：function_codegen.cpp
新增处理：
- RAY_TRACING_INSTANCE_MOTION_SRT → _InstMotionSRT(buffer, instIndex, keyIndex)
- RAY_TRACING_SET_INSTANCE_MOTION_SRT → _SetInstMotionSRT(buffer, instIndex, keyIndex, srt)
- RAY_TRACING_INSTANCE_MOTION_MATRIX → _InstMotionMatrix(buffer, instIndex, keyIndex)
- RAY_TRACING_SET_INSTANCE_MOTION_MATRIX → _SetInstMotionMatrix(buffer, instIndex, keyIndex, mat)

E2. accel_header.bytes 扩展
文件：src/backends/common/hlsl/builtin/accel_header.bytes
新增 HLSL 函数，按 152 字节步长索引 motion instance buffer，偏移到 data union 中的 SRT/Matrix 关键帧位置进行读写。需要处理 VkSRTDataNV ↔ MotionInstanceTransformSRT 的字段重排。

---
阶段 F：集成测试与验证

F1. 编写 VK 专用测试
文件：src/tests/integration/runtime/test_motion_blur_vk.cpp（已存在）
- 简化版测试：仅三角形 mesh + motion instance SRT，不含 curve
- 验证：motion TLAS 构建不崩溃、渲染结果与 CUDA 后端对比

F2. 分阶段验证点

| 阶段 | 验证方法 |
|------|----------|
| A 完成后 | TLAS 构建成功（validation layer 无错误），但渲染结果为静态（time=0） |
| B 完成后 | 顶点运动模糊的 BLAS 构建成功 |
| C+D 完成后 | motion blur trace 正确传递 time，渲染结果出现模糊效果 |
| E 完成后 | shader 中可读写 motion instance 关键帧 |

---
2. 工作量估算与优先级

| 阶段 | 预估工作量 | 风险 | 优先级 |
|------|------------|------|--------|
| A. TLAS Motion Instance | 3-4 天 | 低 | P0 |
| B. BLAS 顶点运动模糊 | 2-3 天 | 低 | P1 |
| C. RT Pipeline 基础设施 | 5-8 天 | 高 | P0 |
| D. Motion Blur Codegen | 2-3 天 | 中 | P0（依赖 C） |
| E. Instance Motion 读写 | 1-2 天 | 低 | P2 |
| F. 集成测试 | 2-3 天 | 中 | P0 |

总计约 15-23 个工作日。

阶段 C（RT Pipeline）是关键路径，建议与阶段 A 并行开发。