#pragma once

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rtx/accel.h>

namespace luisa::compute::hip {

class HIPDevice;
class HIPCommandEncoder;

class HIPAccel {

public:
    struct Binding {
        uint64_t handle;
        uint64_t instance_buffer;
    };

    // Codegen-visible per-instance data. Must match the LLVM accel_instance_type layout:
    //   { [3 x <4 x float>] affine, u32 user_id, u32 sbt_offset, u32 mask, u32 flags, u64 handle }
    struct alignas(16) CodegenInstance {
        float affine[3][4];// row-major 3x4
        uint32_t user_id;
        uint32_t sbt_offset;
        uint32_t visibility_mask;
        uint32_t flags;
        uint64_t mesh_handle;
    };

private:
    AccelOption _option;
    hiprtContext _hiprt_ctx{nullptr};
    hiprtScene _scene{nullptr};
    bool _requires_rebuild{true};
    mutable spin_mutex _mutex;

    hipDeviceptr_t _instance_buffer{};
    size_t _instance_buffer_size{};

    luisa::vector<CodegenInstance> _host_instances;
    luisa::vector<hiprtInstance> _hiprt_instances;
    luisa::vector<hiprtFrameMatrix> _hiprt_frames;

    size_t _instance_count{};

    void _build(HIPCommandEncoder &encoder) noexcept;
    void _update(HIPCommandEncoder &encoder) noexcept;

public:
    explicit HIPAccel(hiprtContext ctx, const AccelOption &option) noexcept;
    ~HIPAccel() noexcept;
    void build(HIPCommandEncoder &encoder, AccelBuildCommand *command) noexcept;
    [[nodiscard]] Binding binding() const noexcept;
};

}// namespace luisa::compute::hip
