#include <luisa/tensor/fallback/softmax_impl.h>
#include <luisa/dsl/sugar.h>
namespace luisa::compute {
namespace softmax_detail {
template<typename T>
struct DispatchPack {
    Kernel1D<void(Buffer<T>)> kernel;
    uint dispatch_size;
};
template<typename T>
Kernel1D<void(Buffer<T>, Buffer<T>, uint, bool)> batch_softmax_kernel() {
    auto batch = Kernel1D([=](BufferVar<T> input, BufferVar<T> output, UInt size, Bool compute_exp) {
        auto block_size = 1024;
        set_block_size(block_size, 1, 1);
        Shared<float> shared_arr(block_size);
        auto thd_id = thread_id().x;
        Float value;
        auto id = dispatch_id().x;
        $if (id < size) {
            $if (compute_exp) {
                value = exp(Float(input.read(id)));
            }
            $else {
                value = Float(input.read(id));
            };
        }
        $else {
            value = 0.0f;
        };
        shared_arr[thd_id] = value;
        UInt thd_size = block_size / 2u;
        sync_block();
        $while (thd_size > 0) {
            $if (thd_id < thd_size) {
                value = shared_arr[thd_id * 2] + shared_arr[thd_id * 2 + 1];
            };
            sync_block();
            $if (thd_id < thd_size) {
                shared_arr[thd_id] = value;
            };
            thd_size /= 2u;
            sync_block();
        };
        $if (thd_id == 0) {
            output.write(block_id().x, Var<T>{shared_arr[0]});
        };
    });

    return batch;
}
template<typename T>
Kernel1D<void(Buffer<T>, Buffer<T>)> batch_final_kernel() {
    auto final = Kernel1D([=](BufferVar<T> buffer, BufferVar<T> sum_buffer) {
        auto id = dispatch_id().x;
        buffer.write(id, Var<T>{exp(Float{buffer.read(id)}) / Float{sum_buffer.read(0u)}});
    });
    return final;
}

template<typename T>
DispatchPack<T> softmax_kernel(uint size) {
    if (size > 1024) {
        LUISA_ERROR("Softmax size can not be larger than 2048");
    }
    if (size == 0u) {
        LUISA_ERROR("Softmax size can not be 0");
    }
    auto block_size = next_pow2(size);
    block_size = std::max<uint>(block_size, 32u);
    auto kernel = Kernel1D([=](BufferVar<T> input) {
        set_block_size(block_size, 1, 1);
        Shared<float> shared_arr(block_size);
        auto thd_id = thread_id().x;
        Float value;
        auto id = dispatch_id().x;
        $if (id < size) {
            value = exp(Float(input.read(id)));
        }
        $else {
            value = 0.0f;
        };
        shared_arr[thd_id] = value;
        UInt thd_size = block_size / 2u;
        sync_block();
        $while (thd_size > 0) {
            $if (thd_id < thd_size) {
                value = shared_arr[thd_id * 2] + shared_arr[thd_id * 2 + 1];
            };
            sync_block();
            $if (thd_id < thd_size) {
                shared_arr[thd_id] = value;
            };
            thd_size /= 2u;
            sync_block();
        };
        $if (id < size) {
            auto write_id = id;
            input.write(write_id, (exp(Float(input.read(write_id))) / shared_arr[0]).template cast<T>());
        };
    });
    return DispatchPack{
        .kernel = std::move(kernel),
        .dispatch_size = (size + block_size - 1u) & (~(block_size - 1u))};
}
}// namespace softmax_detail
SoftmaxImpl::SoftmaxImpl(
    DeviceInterface *device,
    ShaderManager *shader_manager,
    SoftmaxExpr *expr)
    : expr(expr),
      shaders([&]() -> luisa::variant<
                        LargeBatchShader,
                        ShaderManager::ShaderDispatch> {
          struct Tuple {
              TensorElementType ele_type;
              uint dispatch_size;
              uint user_id;
          };
          Tuple tp{
              expr->input->element_type()};
          if (expr->input->get_size(0) <= 1024) {
              tp.user_id = 0;
              tp.dispatch_size = expr->input->get_size(0);
              auto elem_type = expr->input->element_type();
              auto gen_softmax_shader = [&]<typename T>() {
                  auto kernel = softmax_detail::softmax_kernel<T>(expr->input->get_size(0));
                  auto shader = device->create_shader(ShaderOption{}, Function{kernel.kernel.function().get()});
                  return ShaderManager::ShaderDispatch{
                      shader.handle,
                      uint3(kernel.dispatch_size, 1, 1),
                      ShaderDispatchCmdEncoder::compute_uniform_size(kernel.kernel.function()->unbound_arguments())};
              };
              auto ptr = shader_manager->add_shader(
                  TensorExpr::Tag::ESoftmaxExpr,
                  vstd::MD5{{reinterpret_cast<uint8_t const *>(&tp), sizeof(tp)}},
                  [&]() {
                      if (tp.ele_type == TensorElementType::Float16) {
                          return gen_softmax_shader.template operator()<half>();
                      } else if (tp.ele_type == TensorElementType::Float32) {
                          return gen_softmax_shader.template operator()<float>();
                      } else {
                          LUISA_ERROR("Currently softmax can not support this type.");
                      }
                  });
              return ptr;
          } else {
              tp.user_id = 1;
              auto batch_ptr = shader_manager->add_shader(
                  TensorExpr::Tag::ESoftmaxExpr,
                  vstd::MD5{{reinterpret_cast<uint8_t const *>(&tp), sizeof(tp)}},
                  [&]() {
                      ShaderCreationInfo shader;
                      Function func;
                      if (tp.ele_type == TensorElementType::Float16) {
                          auto kernel = softmax_detail::batch_softmax_kernel<half>();
                          shader = device->create_shader(ShaderOption{}, Function{kernel.function().get()});
                          func = Function{kernel.function().get()};
                      } else if (tp.ele_type == TensorElementType::Float32) {
                          auto kernel = softmax_detail::batch_softmax_kernel<float>();
                          shader = device->create_shader(ShaderOption{}, Function{kernel.function().get()});
                          func = Function{kernel.function().get()};
                      } else {
                          LUISA_ERROR("Currently softmax can not support this type.");
                      }
                      return ShaderManager::ShaderDispatch{
                          shader.handle,
                          uint3(1, 1, 1),
                          ShaderDispatchCmdEncoder::compute_uniform_size(func.unbound_arguments())};
                  });
              tp.user_id = 2;
              auto final_ptr = shader_manager->add_shader(
                  TensorExpr::Tag::ESoftmaxExpr,
                  vstd::MD5{{reinterpret_cast<uint8_t const *>(&tp), sizeof(tp)}},
                  [&]() {
                      ShaderCreationInfo shader;
                      Function func;
                      if (tp.ele_type == TensorElementType::Float16) {
                          auto kernel = softmax_detail::batch_final_kernel<half>();
                          shader = device->create_shader(ShaderOption{}, Function{kernel.function().get()});
                          func = Function{kernel.function().get()};
                      } else if (tp.ele_type == TensorElementType::Float32) {
                          auto kernel = softmax_detail::batch_final_kernel<float>();
                          shader = device->create_shader(ShaderOption{}, Function{kernel.function().get()});
                          func = Function{kernel.function().get()};
                      } else {
                          LUISA_ERROR("Currently softmax can not support this type.");
                      }
                      return ShaderManager::ShaderDispatch{
                          shader.handle,
                          uint3(1, 1, 1),
                          ShaderDispatchCmdEncoder::compute_uniform_size(func.unbound_arguments())};
                  });
              return LargeBatchShader{batch_ptr, final_ptr};
          }
      }()) {
}
SoftmaxImpl::~SoftmaxImpl() {
}
void SoftmaxImpl::execute(FallbackTensorCallback *callback, CommandList &cmdlist) const {
}
}// namespace luisa::compute