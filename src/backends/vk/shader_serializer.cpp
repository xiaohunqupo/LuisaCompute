#include "shader_serializer.h"
#include "shader.h"
#include "../common/hlsl/shader_property.h"
#include "compute_shader.h"

namespace lc::vk {
namespace detail {
struct ShaderSerHeader {
    uint64 header_ver;
    vstd::MD5 md5;
    vstd::MD5 type_md5;
    uint64 property_size;
    uint64 spv_byte_size;
    uint block_size[3];
    uint kernel_arg_count;
    bool use_bindless_buffer;
    bool use_bindless_tex2d;
    bool use_bindless_tex3d;
};
struct PSODataPackage {
    VkPipelineCacheHeaderVersionOne header;
    std::byte md5[sizeof(vstd::MD5)];
};
}// namespace detail
void ShaderSerializer::serialize_bytecode(
    vstd::span<const hlsl::Property> binds,
    vstd::span<const SavedArgument> saved_args,
    vstd::MD5 shader_md5,
    vstd::MD5 type_md5,
    uint3 block_size,
    vstd::string_view file_name,
    vstd::span<const uint> spv_code,
    SerdeType serde_type,
    BinaryIO const *bin_io,
    bool use_tex2d_bindless,
    bool use_tex3d_bindless,
    bool use_buffer_bindless) {
    using namespace detail;
    vstd::vector<std::byte> results;
    ShaderSerHeader header{
        .md5 = shader_md5,
        .type_md5 = type_md5,
        .block_size = {block_size.x, block_size.y, block_size.z},
        .property_size = binds.size(),
        .spv_byte_size = spv_code.size_bytes(),
        .kernel_arg_count = (uint)saved_args.size()};
    uint64_t final_size = sizeof(ShaderSerHeader) + header.property_size * sizeof(hlsl::Property) + header.spv_byte_size + header.kernel_arg_count * sizeof(SavedArgument);
    results.push_back_uninitialized(final_size);
    auto data_ptr = results.data();
    auto save = [&]<typename T>(T const &t) {
        memcpy(data_ptr, &t, sizeof(T));
        data_ptr += sizeof(T);
    };
    auto save_arr = [&]<typename T>(T const *t, size_t size) {
        memcpy(data_ptr, t, sizeof(T) * size);
        data_ptr += sizeof(T) * size;
    };
    header.use_bindless_buffer = use_buffer_bindless;
    header.use_bindless_tex2d = use_tex2d_bindless;
    header.use_bindless_tex3d = use_tex3d_bindless;
    save(header);
    save_arr(binds.data(), binds.size());
    save_arr(saved_args.data(), saved_args.size());
    save_arr(spv_code.data(), spv_code.size());

    switch (serde_type) {
        case SerdeType::Cache:
            bin_io->write_shader_cache(file_name, results);
            break;
        case SerdeType::Builtin:
            bin_io->write_internal_shader(file_name, results);
            break;
        case SerdeType::ByteCode:
            bin_io->write_shader_bytecode(file_name, results);
            break;
    }
}
void ShaderSerializer::serialize_pso(
    Device *device,
    Shader const *shader,
    vstd::MD5 shader_md5,
    BinaryIO const *bin_io) {
    vstd::vector<std::byte> pso_data;
    if (!shader->serialize_pso(pso_data)) return;
    using namespace detail;
    PSODataPackage package{
        .header = device->pso_header()};
    memcpy(package.md5, &shader_md5, sizeof(vstd::MD5));
    vstd::MD5 pso_md5{
        {reinterpret_cast<uint8_t const *>(&package), sizeof(PSODataPackage)}};
    auto file_name = pso_md5.to_string(false);

    bin_io->write_shader_cache(file_name, pso_data);
}
ShaderSerializer::DeserResult ShaderSerializer::try_deser_compute(
    Device *device,
    // invalid md5 for AOT
    vstd::optional<vstd::MD5> shader_md5,
    vstd::vector<Argument> &&captured,
    vstd::string_view file_name,
    SerdeType serde_type,
    BinaryIO const *bin_io) {
    using namespace detail;
    vstd::vector<hlsl::Property> properties;
    vstd::vector<SavedArgument> saved_args;
    vstd::vector<uint> spv;
    DeserResult result{
        .shader = nullptr};
    uint3 block_size;
    ShaderSerHeader header;
    {
        auto read_stream = [&]() {
            switch (serde_type) {
                case SerdeType::Cache:
                    return bin_io->read_shader_cache(file_name);
                case SerdeType::Builtin:
                    return bin_io->read_internal_shader(file_name);
                case SerdeType::ByteCode:
                    return bin_io->read_shader_bytecode(file_name);
            }
        }();
        if (!read_stream) return result;
        if (read_stream->length() < sizeof(ShaderSerHeader)) return result;
        read_stream->read({reinterpret_cast<std::byte *>(&header), sizeof(ShaderSerHeader)});
        if (read_stream->length() != (header.property_size * sizeof(hlsl::Property) + header.spv_byte_size + header.kernel_arg_count * sizeof(SavedArgument)))
            return result;
        if (shader_md5 && *shader_md5 != header.md5)
            return result;
        result.type_md5 = header.type_md5;
        block_size = uint3(header.block_size[0], header.block_size[1], header.block_size[2]);
        properties.push_back_uninitialized(header.property_size);
        read_stream->read({reinterpret_cast<std::byte *>(properties.data()), properties.size_bytes()});
        saved_args.push_back_uninitialized(header.kernel_arg_count);
        read_stream->read({reinterpret_cast<std::byte *>(saved_args.data()), saved_args.size_bytes()});
        spv.push_back_uninitialized(header.spv_byte_size / sizeof(uint));
        read_stream->read({reinterpret_cast<std::byte *>(spv.data()), spv.size_bytes()});
    }
    vstd::vector<std::byte> pso_data;
    vstd::string pso_name;
    {
        PSODataPackage package{
            .header = device->pso_header()};
        memcpy(package.md5, &shader_md5, sizeof(vstd::MD5));
        vstd::MD5 pso_md5{
            {reinterpret_cast<uint8_t const *>(&package), sizeof(PSODataPackage)}};
        pso_name = pso_md5.to_string(false);
        auto read_stream = bin_io->read_shader_cache(pso_name);
        if (read_stream) {
            if (read_stream->length() >= sizeof(VkPipelineCacheHeaderVersionOne)) {
                pso_data.push_back_uninitialized(sizeof(VkPipelineCacheHeaderVersionOne));
                read_stream->read(pso_data);
                if (!device->is_pso_same(*reinterpret_cast<VkPipelineCacheHeaderVersionOne const *>(pso_data.data()))) {
                    pso_data.clear();
                } else {
                    auto last_size = read_stream->length();
                    pso_data.push_back_uninitialized(last_size);
                    read_stream->read({pso_data.data() + sizeof(VkPipelineCacheHeaderVersionOne), last_size});
                }
            }
        }
    }
    auto shader = new ComputeShader{
        device,
        block_size,
        properties,
        std::move(saved_args),
        spv,
        std::move(captured),
        pso_data,
        header.use_bindless_tex2d,
        header.use_bindless_tex3d,
        header.use_bindless_buffer};
    if (pso_data.empty() &&
        shader->serialize_pso(pso_data)) {
        bin_io->write_shader_cache(pso_name, pso_data);
    }
    result.shader = shader;
    return result;
}
vstd::vector<SavedArgument> ShaderSerializer::serialize_saved_args(Function kernel) {
    LUISA_ASSUME(kernel.tag() != Function::Tag::CALLABLE);
    auto &&args = kernel.arguments();
    vstd::vector<SavedArgument> result;
    vstd::push_back_func(result, args.size(), [&](size_t i) {
        auto &&var = args[i];
        return SavedArgument(kernel, var);
    });
    return result;
}

vstd::vector<SavedArgument> ShaderSerializer::serialize_saved_args(
    vstd::IRange<std::pair<Variable, Usage>> &arguments) {
    vstd::vector<SavedArgument> result;
    for (auto &&i : arguments) {
        result.emplace_back(i.second, i.first);
    }
    return result;
}
}// namespace lc::vk
