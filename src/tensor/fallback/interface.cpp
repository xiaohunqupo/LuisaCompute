#include <luisa/tensor/fallback/interface.h>
#include <luisa/tensor/tensor_builder.h>
#include <luisa/core/first_fit.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/tensor/fallback/gemm_impl.h>

namespace luisa::compute {
FallbackTensorKenel::FallbackTensorKenel(DeviceInterface* device, luisa::unique_ptr<TensorBuilder> &&t_args)
    :device(device), tensor_builder(std::move(t_args)) {
    expr_topo.init(tensor_builder->root_expr().expressions, tensor_builder->allocated_tensor().size());
    auto sorted_tensors = expr_topo.topo_sort();
    executors.reserve(sorted_tensors.size());
    FirstFit first_fit(1ull << 48ull, 16ull);
    tensor_sub_nodes.resize(tensor_builder->allocated_tensor().size());
    luisa::vector<std::pair<FirstFit::Node *, size_t>> allocated_nodes(tensor_sub_nodes.size());
    luisa::vector<FirstFit::Node *> remove_nodes;
    for (auto &i : tensor_builder->arguments()) {
        allocated_nodes[i->idx()].second = std::numeric_limits<size_t>::max();
    }
    // Accumulate ref-count
    size_t buffer_size_bytes = 0;
    for (auto &i : sorted_tensors) {
        auto func = [&](TensorData *tensor_data, Usage usage) {
            auto &node = allocated_nodes[tensor_data->idx()];
            // This is an input tensor, no need allocate
            if (node.second == std::numeric_limits<size_t>::max()) {
                return;
            }
            node.second++;
        };
        i->get_tensors(func);
    }
    // Make allocate
    for (auto &i : sorted_tensors) {
        auto func = [&](TensorData *tensor_data, Usage usage) {
            auto &node = allocated_nodes[tensor_data->idx()];
            // This is an input tensor, no need allocate
            if (node.second == std::numeric_limits<size_t>::max()) {
                return;
            }
            if (!node.first) {
                node.first = first_fit.allocate_best_fit(tensor_data->size_bytes());
                if (!node.first) [[unlikely]] {
                    LUISA_ERROR("Tensor graph too large, allocate failed.");
                }
                buffer_size_bytes = std::max(buffer_size_bytes, node.first->offset() + node.first->size());
                tensor_sub_nodes[tensor_data->idx()] = node.first->offset();
            }
            node.second--;
            if (node.second == 0) {
                remove_nodes.emplace_back(node.first);
            }
        };
        i->get_tensors(func);
        for (auto &i : remove_nodes) {
            first_fit.free(i);
        }
        remove_nodes.clear();
        if (buffer_size_bytes > 0)
            tensor_buffer = device->create_buffer(Type::of<float4>(), buffer_size_bytes / sizeof(float4), nullptr);
        else
            tensor_buffer.invalidate();
        // Make executors
        switch (i->tag()) {
            case TensorExpr::Tag::EGEMMExpr: {
                executors.emplace_back(luisa::make_unique<GemmImpl>(device, static_cast<GEMMExpr *>(i)));
            } break;
            default: {
                LUISA_ERROR("Expr not supported.");
            } break;
        }
    }
}
FallbackTensorKenel::~FallbackTensorKenel() {
    if (tensor_buffer.handle != invalid_resource_handle)
        device->destroy_buffer(tensor_buffer.handle);
}
void FallbackTensorKenel::check(luisa::span<Argument::Buffer const> tensors) {
    auto args = tensor_builder->arguments();
    if (tensors.size() != args.size()) {
        LUISA_ERROR("Argument size dismatch.");
    }
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i]->size_bytes() != tensors[i].size) {
            LUISA_ERROR("Argument {} size {} dismatch with dest {}.", i, tensors[i].size, args[i]->size_bytes());
        }
        size_t align = tensor_element_align(args[i]->element_type());
        if ((tensors[i].offset & (align - 1)) != 0) {
            LUISA_ERROR("Argument {} with offset {} should be align to {}", i, tensors[i].offset, align);
        }
    }
}
void *FallbackTensorInterface::compile_kernel(luisa::unique_ptr<TensorBuilder> &&tensor_builder) noexcept {
    return luisa::new_with_allocator<FallbackTensorKenel>(device(), std::move(tensor_builder));
}
void FallbackTensorInterface::destroy_kernel(void *kernel_ptr) noexcept {
    luisa::delete_with_allocator(static_cast<FallbackTensorKenel *>(kernel_ptr));
}
Argument::Buffer FallbackTensorCallback::get_tensor_buffer(TensorData *data) {
    auto iter = args->find(data);
    // is arg
    if (iter != args->end()) {
        return iter->second;
    }
    return Argument::Buffer{
        .handle = kernel->tensor_buffer.handle,
        .offset = kernel->tensor_sub_nodes[data->idx()],
        .size = data->size_bytes()};
}
void FallbackTensorInterface::execute(
    CommandList &cmdlist,
    void *kernel_ptr,
    luisa::span<Argument::Buffer const> tensors) noexcept {
    auto kernel = static_cast<FallbackTensorKenel *>(kernel_ptr);
    kernel->device = device();
    kernel->check(tensors);
    luisa::unordered_map<TensorData *, Argument::Buffer> arguments;
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
        arguments.try_emplace(kernel->tensor_builder->arguments()[idx], tensors[idx]);
    }
    FallbackTensorCallback callback;
    callback.args = &arguments;
    callback.kernel = kernel;
    for (auto &i : kernel->executors) {
        i->execute(&callback, cmdlist);
    }
}

LC_TENSOR_API luisa::unique_ptr<TensorInterface> TensorInterface::create_fallback_backend(Device& device) {
    return luisa::make_unique<FallbackTensorInterface>(device);
}
}// namespace luisa::compute