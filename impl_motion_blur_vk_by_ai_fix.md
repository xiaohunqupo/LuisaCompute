## 问题 1：了解实现原理
已通读文档和代码。核心架构是双路径：
- 非 motion blur：Compute + Inline Ray Query（RayQuery<>::TraceRayInline()）
- Motion blur：Ray Tracing Pipeline（TraceRay() → SPIR-V 后处理为 OpTraceRayMotionNV）

## 问题 2：test_mesh_3 黑图修复
根因有两个：
1. Motion instance stride 未对齐 16 字节：VkAccelerationStructureMotionInstanceNV 的 C struct 大小是 152 字节，但 Vulkan 驱动要求 motion instance 在 instance buffer 中 16 字节对齐。152 不是 16 的倍数，导致多实例场景下第 2 个及之后的 instance 地址未对齐，驱动只能识别第 1 个实例。修复：stride 改为 ceil(152/16)*16 = 160。
2. Instance buffer stride 不匹配：motion 模式下 TLAS 构建用 160 字节 stride 的 buffer，但 shader 通过 StructuredBuffer<_MeshInst>（64 字节 stride）读取 instance 数据（如 instance_user_id）。单 buffer 时，shader 读取第 N 个 instance 会读到错误偏移。修复：拆分为两个 buffer——_motion_instance_buffer（160 字节 stride，TLAS 构建用）和 _instance_buffer（64 字节 stride，shader 读取用）。
改动文件：
- src/backends/vk/tlas.h — 新增 _motion_instance_buffer 和 kMotionInstanceStride = 160
- src/backends/vk/tlas.cpp — 拆分 buffer 管理，motion path 同时填充两个 buffer
- src/backends/common/hlsl/builtin/accel_process_vk_motion — stride 从 152 改为 160


## 问题 3：文档 4.3 修正
已更新文档：
- sizeof 是 152 不是 184（文档原始值错误）
- 实际 stride 应为 160（16 字节对齐）
- ByteBuffer alignas(4) 是 HLSL 标准，不是 VK 的
- StructuredBuffer<uint16_t> 是支持的
- vk_config_ext.h 中 enable_motion_blur() 已存在（vk_config_ext.h:46）

## 问题 4：为什么需要后处理生成 OpTraceRayMotionNV
这不是基于某个官方文档的明确要求，而是 DXC 编译器的限制决定的：
1. DXC 不支持 SPV_NV_ray_tracing_motion_blur：DXC（DirectX Shader Compiler）是 HLSL → SPIR-V 的编译器。它支持 SPV_KHR_ray_tracing（生成 OpTraceRayKHR），但不支持 NV 厂商扩展 SPV_NV_ray_tracing_motion_blur。DXC 没有任何 HLSL 语法或内置函数能直接生成 OpTraceRayMotionNV 指令。
2. HLSL 没有 motion blur trace 的标准语法：DirectX 12 的 DXR 1.1 没有 motion blur 的 TraceRay 变体。HLSL 的 TraceRay() 只能生成标准的 OpTraceRayKHR（12 个操作数），没有 time 参数。OpTraceRayMotionNV 需要 13 个操作数（多一个 time）。
3. 后处理方案的工作原理：
   - HLSL 层面：把 time 存入 payload struct 的一个字段，然后调用标准 TraceRay()
   - DXC 编译：生成标准 OpTraceRayKHR + payload 的 OpCompositeConstruct
   - SPIR-V 后处理：找到 OpTraceRayKHR，从 payload 的 OpCompositeConstruct 中提取 time 值，替换为 OpTraceRayMotionNV（插入 time 操作数），注入 capability 和 extension 声明
4. 替代方案不可行：
   - 直接写 SPIR-V：太底层，维护成本高
   - 用 GLSL + glslang：需要重写整个 codegen 管线
   - 等 DXC 支持：NV 扩展不太可能被 DXC 原生支持
简单说：DXC 不认识 OpTraceRayMotionNV，所以只能先让 DXC 生成标准的 OpTraceRayKHR，再在 SPIR-V 二进制层面做 patch。 这是在现有 HLSL→SPIR-V 编译管线约束下的务实选择。

## 参考资料
+ [https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/SPIR-V.rst](https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/SPIR-V.rst)
+ [https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_ray_tracing_motion_blur.html](https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_ray_tracing_motion_blur.html)
