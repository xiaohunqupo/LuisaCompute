#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/clock.h>
#include <luisa/tensor/kernel.h>
#include <luisa/tensor/pass/expr_topo.h>
#include <luisa/tensor/tensor_interface.h>
#include <fstream>
using namespace luisa;
using namespace luisa::compute;

struct MyTensorKenel {
    luisa::unique_ptr<TensorBuilder> tensor_builder;
    MyTensorKenel(luisa::unique_ptr<TensorBuilder> &&tensor_builder) 
    : tensor_builder(std::move(tensor_builder))
    {
    }
    void check(luisa::span<Argument::Buffer const> tensors) {
        auto args = tensor_builder->arguments();
        if(tensors.size() != args.size()) {
            LUISA_ERROR("Argument size dismatch.");
        }
        for(size_t i = 0; i < args.size(); ++i) {
            if(args[i]->size_bytes() != tensors[i].size) {
                LUISA_ERROR("Argument {} size {} dismatch with dest {}.", i, tensors[i].size, args[i]->size_bytes());
            }
            static const size_t aligns[] = {2, 4, 8};
            auto align = aligns[luisa::to_underlying(args[i]->element_type())];

            if((tensors[i].offset & (align - 1)) != 0) {
                LUISA_ERROR("Argument {} with offset {} should be align to {}", i, tensors[i].offset, align);
            }
        }
    }
};

class MyTensorInterface : public TensorInterface {
public:
    MyTensorInterface(Device &device) noexcept : TensorInterface(device) {}
    void *compile_kernel(luisa::unique_ptr<TensorBuilder> &&tensor_builder) noexcept override {
        return new MyTensorKenel(std::move(tensor_builder));
    }
    void destroy_kernel(void *kernel_ptr) noexcept override {
        delete static_cast<MyTensorKenel*>(kernel_ptr);
    }
    void execute(
        CommandList &cmdlist,
        void *kernel_ptr,
        luisa::span<Argument::Buffer const> tensors) noexcept override {
        for (auto &i : tensors) {
            LUISA_INFO("Arg handle {}, offset {}, size {}", i.handle, i.offset, i.size);
        }
        static_cast<MyTensorKenel*>(kernel_ptr)->check(tensors);
    }
};

int main(int argc, char *argv[]) {
    TensorBuilder builder;
    auto ctx = Context(argv[0]);
    auto device = ctx.create_device("dx");
    auto stream = device.create_stream();
    MyTensorInterface interface{device};
    auto kernel = load_tensor(
        &interface,
        [&](Tensor t0,
            Tensor t1) {

        });
    kernel.compile(
        TensorDescriptor{
            std::initializer_list<size_t>{1024, 1ull, 1ull},
            TensorElementType::Float32},
        TensorDescriptor{std::initializer_list<size_t>{1024, 1ull, 1ull}, TensorElementType::Float32});
    auto b0 = device.create_buffer<float>(1024);
    auto b1 = device.create_buffer<float>(1024);
    CommandList cmdlist;
    kernel.execute(cmdlist, b0, b1);
}