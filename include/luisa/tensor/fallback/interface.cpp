#include "interface.h"
#include <luisa/tensor/tensor_builder.h>

namespace luisa::compute {
FallbackTensorKenel::FallbackTensorKenel(luisa::unique_ptr<TensorBuilder> &&tensor_builder)
    : tensor_builder(std::move(tensor_builder)) {
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
        static const size_t aligns[] = {2, 4, 8};
        auto align = aligns[luisa::to_underlying(args[i]->element_type())];

        if ((tensors[i].offset & (align - 1)) != 0) {
            LUISA_ERROR("Argument {} with offset {} should be align to {}", i, tensors[i].offset, align);
        }
    }
}
void *FallbackTensorInterface::compile_kernel(luisa::unique_ptr<TensorBuilder> &&tensor_builder) noexcept {
    return luisa::new_with_allocator<FallbackTensorKenel>(std::move(tensor_builder));
}
void FallbackTensorInterface::destroy_kernel(void *kernel_ptr) noexcept {
    luisa::delete_with_allocator(static_cast<FallbackTensorKenel *>(kernel_ptr));
}
void FallbackTensorInterface::execute(
    CommandList &cmdlist,
    void *kernel_ptr,
    luisa::span<Argument::Buffer const> tensors) noexcept {
    static_cast<FallbackTensorKenel *>(kernel_ptr)->check(tensors);
}
}// namespace luisa::compute