#include "builtin_kernel.h"
#include <luisa/core/stl/filesystem.h>
#include "../common/hlsl/hlsl_codegen.h"
#include "device.h"
namespace lc::vk {
ComputeShader *BuiltinKernel::load_accel_set_kernel(Device *device) {
    auto func = [&] {
        hlsl::CodegenResult code;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.result << hlsl::CodegenUtility::ReadInternalHLSLFile("accel_process_vk");
        code.properties.resize(2);
        auto &set_buffer = code.properties[0];
        set_buffer.array_size = 1;
        set_buffer.register_index = 0;
        set_buffer.space_index = 0;
        set_buffer.type = hlsl::ShaderVariableType::StructuredBuffer;
        auto &inst_buffer = code.properties[1];
        inst_buffer.array_size = 1;
        inst_buffer.register_index = 1;
        inst_buffer.space_index = 0;
        inst_buffer.type = hlsl::ShaderVariableType::RWStructuredBuffer;
        return code;
    };
    vstd::vector<SavedArgument> saved_args;
    return ComputeShader::compile(
        device->binary_io(),
        device,
        std::move(saved_args),
        std::move(func),
        vstd::MD5{"accel_process_vk"sv},
        {},
        uint3(256, 1, 1),
        "accel_process_vk.dxil"sv,
        SerdeType::kBuiltin,
        62, true);
}
ComputeShader *BuiltinKernel::load_accel_motion_set_kernel(Device *device) {
    // Motion variant: uses RWByteAddressBuffer for 184-byte stride output (sizeof(VkAccelerationStructureMotionInstanceNV))
    static constexpr auto motion_shader_source = R"(
struct InputInst{
float4 p0;float4 p1;float4 p2;
uint2 mesh;uint index:24;uint vis_mask:8;uint user_id:24;uint flags:8;
};
RWByteAddressBuffer _InstBuffer:register(u1);
StructuredBuffer<InputInst> _SetBuffer:register(t0);
struct _CBType{uint dsp;uint count;};
[[vk::push_constant]] ConstantBuffer<_CBType> cb:register(b0);
static const uint MOTION_INST_STRIDE = 184u;
static const uint STATIC_DATA_OFFSET = 8u;
[numthreads(256,1,1)]
void main(uint id:SV_DISPATCHTHREADID){
if(id >= cb.dsp) return;
const uint flag_mesh=1u << 0u;
const uint flag_transform=1u << 1u;
const uint flag_opaque_on = 1u << 2u;
const uint flag_opaque_off = 1u << 3u;
const uint flag_visibility = 1u << 4u;
const uint flag_user_id = 1u << 5u;
const uint flag_opaque = flag_opaque_on | flag_opaque_off;
InputInst v=_SetBuffer[id];
if(v.index>=cb.count) return;
uint base = v.index * MOTION_INST_STRIDE;
_InstBuffer.Store(base, 0u);
_InstBuffer.Store(base + 4u, 0u);
uint db = base + STATIC_DATA_OFFSET;
if((v.flags&flag_transform)!=0){
_InstBuffer.Store4(db + 0u, asuint(v.p0));
_InstBuffer.Store4(db + 16u, asuint(v.p1));
_InstBuffer.Store4(db + 32u, asuint(v.p2));
}
if((v.flags&flag_visibility)!=0 || (v.flags&flag_user_id)!=0){
uint e = _InstBuffer.Load(db + 48u);
uint iid = e & 0x00FFFFFFu;
uint imk = (e >> 24u) & 0xFFu;
if((v.flags&flag_user_id)!=0) iid = v.user_id;
if((v.flags&flag_visibility)!=0) imk = v.vis_mask;
_InstBuffer.Store(db + 48u, iid | (imk << 24u));
}
{
uint hf = 0u;
if((v.flags&flag_opaque)!=0){
if((v.flags&flag_opaque_on)!=0) hf = 4u; else hf = 8u;
}
_InstBuffer.Store(db + 52u, 0u | (hf << 24u));
}
if((v.flags&flag_mesh)!=0){
_InstBuffer.Store2(db + 56u, v.mesh);
}
}
)"sv;
    auto func = [&] {
        hlsl::CodegenResult code;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.result << motion_shader_source;
        code.properties.resize(2);
        auto &set_buffer = code.properties[0];
        set_buffer.array_size = 1;
        set_buffer.register_index = 0;
        set_buffer.space_index = 0;
        set_buffer.type = hlsl::ShaderVariableType::StructuredBuffer;
        auto &inst_buffer = code.properties[1];
        inst_buffer.array_size = 1;
        inst_buffer.register_index = 1;
        inst_buffer.space_index = 0;
        inst_buffer.type = hlsl::ShaderVariableType::RWStructuredBuffer;
        return code;
    };
    vstd::vector<SavedArgument> saved_args;
    return ComputeShader::compile(
        device->binary_io(),
        device,
        std::move(saved_args),
        std::move(func),
        vstd::MD5{"accel_process_vk_motion"sv},
        {},
        uint3(256, 1, 1),
        "accel_process_vk_motion.spv"sv,
        SerdeType::kCache,
        62, true);
}
ComputeShader *BuiltinKernel::load_bindless_set_kernel(Device *device) {
    auto func = [&] {
        hlsl::CodegenResult code;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.result << hlsl::CodegenUtility::ReadInternalHLSLFile("bindless_upload_vk");
        code.properties.resize(2);
        auto &set_buffer = code.properties[0];
        set_buffer.array_size = 1;
        set_buffer.register_index = 0;
        set_buffer.space_index = 0;
        set_buffer.type = hlsl::ShaderVariableType::StructuredBuffer;
        auto &inst_buffer = code.properties[1];
        inst_buffer.array_size = 1;
        inst_buffer.register_index = 1;
        inst_buffer.space_index = 0;
        inst_buffer.type = hlsl::ShaderVariableType::RWStructuredBuffer;
        return code;
    };
    vstd::vector<SavedArgument> saved_args;
    return ComputeShader::compile(
        device->binary_io(),
        device,
        std::move(saved_args),
        std::move(func),
        vstd::MD5{"bindless_upload_vk"sv},
        {},
        uint3(256, 1, 1),
        "load_bdls_vk.dxil"sv,
        SerdeType::kBuiltin,
        62, true);
}
// namespace detail {
// static ComputeShader *LoadBCKernel(
//     Device *device,
//     vstd::function<vstd::string_view()> const &includeCode,
//     vstd::function<vstd::string_view()> const &kernelCode,
//     vstd::string_view codePath) {
//     auto func = [&] {
//         hlsl::CodegenResult code;
//         auto incCode = includeCode();
//         auto kerCode = kernelCode();
//         code.result.reserve(incCode.size() + kerCode.size());
//         code.result << incCode << kerCode;
//         code.useBufferBindless = false;
//         code.useTex2DBindless = false;
//         code.useTex3DBindless = false;
//         code.properties.resize(4);
//         auto &globalBuffer = code.properties[0];
//         globalBuffer.array_size = 1;
//         globalBuffer.register_index = 0;
//         globalBuffer.space_index = 0;
//         globalBuffer.type = hlsl::ShaderVariableType::ConstantBuffer;

//         auto &gInput = code.properties[1];
//         gInput.array_size = 1;
//         gInput.register_index = 0;
//         gInput.space_index = 0;
//         gInput.type = hlsl::ShaderVariableType::SRVTextureHeap;

//         auto &gInBuff = code.properties[2];
//         gInBuff.array_size = 1;
//         gInBuff.register_index = 1;
//         gInBuff.space_index = 0;
//         gInBuff.type = hlsl::ShaderVariableType::StructuredBuffer;

//         auto &gOutBuff = code.properties[3];
//         gOutBuff.array_size = 1;
//         gOutBuff.register_index = 0;
//         gOutBuff.space_index = 0;
//         gOutBuff.type = hlsl::ShaderVariableType::RWStructuredBuffer;
//         return code;
//     };
//     vstd::string fileName;
//     vstd::string_view extName = "2.dxil"sv;
//     fileName.reserve(codePath.size() + extName.size());
//     fileName << codePath << extName;
//     return ComputeShader::CompileCompute(
//         device->fileIo,
//         device->profiler,
//         device,
//         {},
//         func,
//         {},
//         {},
//         uint3(1, 1, 1),
//         62,
//         fileName,
//         CacheType::Internal, true, false);
// }
// static vstd::string_view Bc6Header() {
//     static auto bc6Header = hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_header");
//     return {bc6Header.data(), bc6Header.size()};
// }
// static vstd::string_view Bc7Header() {
//     static auto bc7Header = hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_header");
//     return {bc7Header.data(), bc7Header.size()};
// }

// static vstd::string bc7Header;
// }// namespace detail

// ComputeShader *BuiltinKernel::LoadBC6TryModeG10CSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc6Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_trymode_g10cs"); },
//         "bc6_trymodeg10"sv);
// }
// ComputeShader *BuiltinKernel::LoadBC6TryModeLE10CSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc6Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_trymode_le10cs"); },
//         "bc6_trymodele10"sv);
// }
// ComputeShader *BuiltinKernel::LoadBC6EncodeBlockCSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc6Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_encode_block"); },
//         "bc6_encodeblock"sv);
// }
// ComputeShader *BuiltinKernel::LoadBC7TryMode456CSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc7Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_trymode_456cs"); },
//         "bc7_trymode456"sv);
// }
// ComputeShader *BuiltinKernel::LoadBC7TryMode137CSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc7Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_trymode_137cs"); },
//         "bc7_trymode137"sv);
// }
// ComputeShader *BuiltinKernel::LoadBC7TryMode02CSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc7Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_trymode_02cs"); },
//         "bc7_trymode02"sv);
// }
// ComputeShader *BuiltinKernel::LoadBC7EncodeBlockCSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc7Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_encode_block"); },
//         "bc7_encodeblock"sv);
// }
}// namespace lc::vk
