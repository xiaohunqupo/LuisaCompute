#include <luisa/core/magic_enum.h>
#include <Shader/Shader.h>
#include <d3dcompiler.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <Resource/TopAccel.h>
#include <Resource/DefaultBuffer.h>
#include <Shader/ShaderSerializer.h>
#include <luisa/core/logging.h>

namespace lc::dx {
SavedArgument::SavedArgument(Usage usage, Variable const &var)
    : SavedArgument(var.type()) {
    varUsage = usage;
}

SavedArgument::SavedArgument(Function kernel, Variable const &var)
    : SavedArgument(var.type()) {
    varUsage = kernel.variable_usage(var.uid());
}
SavedArgument::SavedArgument(Type const *type) {
    if (luisa::to_underlying(type->tag()) < luisa::to_underlying(Type::Tag::BUFFER)) {
        structSize = type->size();
    }
}

Shader::Shader(
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ComPtr<ID3D12RootSignature> &&rootSig,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers)
    : rootSig(std::move(rootSig)),
      _properties(std::move(prop)),
      kernelArguments(std::move(args)),
      _printers(std::move(printers)) {
}

Shader::Shader(
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ID3D12Device *device,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    bool isRaster)
    : _properties(std::move(prop)),
      kernelArguments(std::move(args)),
      _printers(std::move(printers)) {
    auto serializedRootSig = ShaderSerializer::SerializeRootSig(
        _properties,
        isRaster);
    ThrowIfFailed(device->CreateRootSignature(
        0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(rootSig.GetAddressOf())));
}

void Shader::set_compute_resource(
    uint propertyName,
    CommandBufferBuilder *cb,
    BufferView buffer) const {
    auto cmdList = cb->get_cb()->cmd_list();
    auto &&var = _properties[propertyName];
    switch (var.type) {
        case hlsl::ShaderVariableType::ConstantBuffer: {
            cmdList->SetComputeRootConstantBufferView(
                propertyName,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::StructuredBuffer: {
            cmdList->SetComputeRootShaderResourceView(
                propertyName,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::RWStructuredBuffer: {
            cmdList->SetComputeRootUnorderedAccessView(
                propertyName,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        default:
            LUISA_ERROR("Invalid shader resource type {}.\n\n"
                        "This might be due to the change of shader cache.\n"
                        "Please try delete the cache folder (default: build_dir/bin/.cache) "
                        "and re-run the program.\n"
                        "If the problem persists, please report this issue to the developers.\n",
                        to_string(var.type));
    }
}
void Shader::set_compute_resource(
    uint propertyName,
    CommandBufferBuilder *cb,
    DescriptorHeapView view) const {
    auto cmdList = cb->get_cb()->cmd_list();
    auto &&var = _properties[propertyName];
    switch (var.type) {
        case hlsl::ShaderVariableType::UAVBufferHeap:
        case hlsl::ShaderVariableType::UAVTextureHeap:
        case hlsl::ShaderVariableType::CBVBufferHeap:
        case hlsl::ShaderVariableType::SamplerHeap:
        case hlsl::ShaderVariableType::SRVBufferHeap:
        case hlsl::ShaderVariableType::SRVTextureHeap: {
            cmdList->SetComputeRootDescriptorTable(
                propertyName,
                view.heap->hGPU(view.index));
        } break;
        default: LUISA_ASSUME(false); break;
    }
}
void Shader::set_compute_resource(
    uint propertyName,
    CommandBufferBuilder *cb,
    std::pair<uint, uint4> const &constValue) const {
    auto cmdList = cb->get_cb()->cmd_list();
    LUISA_ASSUME(_properties[propertyName].type == hlsl::ShaderVariableType::ConstantValue);
    cmdList->SetComputeRoot32BitConstants(propertyName, constValue.first, &constValue.second, 0);
}
void Shader::set_compute_resource(
    uint propertyName,
    CommandBufferBuilder *cmdList,
    TopAccel const *bAccel) const {
    return set_compute_resource(
        propertyName,
        cmdList,
        BufferView(bAccel->GetAccelBuffer()));
}
void Shader::set_raster_resource(
    uint propertyName,
    CommandBufferBuilder *cb,
    BufferView buffer) const {
    auto cmdList = cb->get_cb()->cmd_list();
    auto &&var = _properties[propertyName];
    switch (var.type) {
        case hlsl::ShaderVariableType::ConstantBuffer: {
            cmdList->SetGraphicsRootConstantBufferView(
                propertyName,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::StructuredBuffer: {
            cmdList->SetGraphicsRootShaderResourceView(
                propertyName,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::RWStructuredBuffer: {
            cmdList->SetGraphicsRootUnorderedAccessView(
                propertyName,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        default: LUISA_ASSUME(false); break;
    }
}
void Shader::set_raster_resource(
    uint propertyName,
    CommandBufferBuilder *cb,
    DescriptorHeapView view) const {
    auto cmdList = cb->get_cb()->cmd_list();
    auto &&var = _properties[propertyName];
    switch (var.type) {
        case hlsl::ShaderVariableType::UAVBufferHeap:
        case hlsl::ShaderVariableType::UAVTextureHeap:
        case hlsl::ShaderVariableType::CBVBufferHeap:
        case hlsl::ShaderVariableType::SamplerHeap:
        case hlsl::ShaderVariableType::SRVBufferHeap:
        case hlsl::ShaderVariableType::SRVTextureHeap: {
            cmdList->SetGraphicsRootDescriptorTable(
                propertyName,
                view.heap->hGPU(view.index));
        } break;
        default: LUISA_ASSUME(false); break;
    }
}
void Shader::set_raster_resource(
    uint propertyName,
    CommandBufferBuilder *cmdList,
    TopAccel const *bAccel) const {
    return set_raster_resource(
        propertyName,
        cmdList,
        BufferView(bAccel->GetAccelBuffer()));
}
void Shader::set_raster_resource(
    uint propertyName,
    CommandBufferBuilder *cb,
    std::pair<uint, uint4> const &constValue) const {
    auto cmdList = cb->get_cb()->cmd_list();
    LUISA_ASSUME(_properties[propertyName].type == hlsl::ShaderVariableType::ConstantValue);
    cmdList->SetGraphicsRoot32BitConstants(propertyName, constValue.first, &constValue.second, 0);
}
void Shader::save_pso(ID3D12PipelineState *pso, vstd::string_view psoName, luisa::BinaryIO const *fileStream, Device const *device) const {
    LUISA_VERBOSE("Write Pipeline cache to {}.", psoName);
    ComPtr<ID3DBlob> psoCache;
    pso->GetCachedBlob(&psoCache);
    static_cast<void>(fileStream->write_shader_cache(
        psoName,
        {reinterpret_cast<std::byte const *>(psoCache->GetBufferPointer()),
         psoCache->GetBufferSize()}));
};
vstd::string Shader::pso_name(Device const *device, vstd::string_view fileName) {
    vstd::fixed_vector<uint8_t, 64> data;
    luisa::enlarge_by(data, 16 + fileName.size());
    std::memcpy(data.data(), &device->adapter_id, 16);
    std::memcpy(data.data() + 16, fileName.data(), fileName.size());
    vstd::MD5 hash{data};
    return hash.to_string(false) + ".dx";
}

}// namespace lc::dx
