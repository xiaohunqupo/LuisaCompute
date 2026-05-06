## 改动总结
1. src/backends/vk/tlas.cpp — 实现 SRT Motion Instance 写入
核心改动在 tlas.cpp:178-237 的 motion instance buffer 填充循环中。之前所有 motion instance 都写入 type=0 (STATIC)，SRT 关键帧数据被完全忽略。
改动内容：
- 在填充循环中检测每个 modification 对应的 primitive 是否为 MotionInstance
- 当检测到 SRT 模式的 MotionInstance 时，写入 type=VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_SRT_MOTION_NV (2)
- 实现 MotionInstanceTransformSRT → VkSRTDataNV 的字段重排（与 CUDA 后端 cuda_motion_instance.cpp:105-124 一致）
- 填充 transformT0（第一个关键帧）和 transformT1（最后一个关键帧）
- 当关键帧数 > 2 时取首尾两帧并输出 warning（Vulkan NV 扩展限制）
- 非 MotionInstance 的实例仍然走 STATIC 路径
2. src/tests/integration/runtime/motion_blur/test_mesh_4.cpp — 新测试用例
测试 Motion Instance SRT 变换：
- 创建一个静态三角形 mesh（无顶点动画）
- 用 MotionInstance 包裹，设置 SRT 模式、2 个关键帧
- 关键帧 0：无旋转（0°），关键帧 1：绕 Y 轴旋转 45°
- 使用 accel.intersect_motion(ray, time, {}) 进行带时间参数的光线追踪
- 1024 spp 渐进式累积渲染
3. src/tests/CMakeLists.txt — 注册测试
添加 test_mesh_4 构建目标。

## 验证结果
- 编译通过
- 运行 ./build/debug/bin/test_mesh_4 vk 成功
- SPIR-V patcher 正确替换了 1 个 OpTraceRayKHR → OpTraceRayMotionNV
- 输出图片 test_motion_blur_mesh_4.png 显示一个带有旋转运动模糊的三角形（y=182~338, 边缘有模糊过渡）