#include <luisa/tensor/tensor.h>
#include <luisa/tensor/tensor_builder.h>
#include <luisa/tensor/expression.h>
#include <luisa/core/logging.h>

namespace luisa::compute {
/////////////// Static assert
#define LUISA_TENSOR_EXPR_STATIC_ASSERT(CMD) static_assert(std::is_trivially_destructible_v<CMD>, "Expr type must be trivially destructible.");
LUISA_MAP(LUISA_TENSOR_EXPR_STATIC_ASSERT, LUISA_COMPUTE_TENSOR_EXPRRESSIONS)
#undef LUISA_TENSOR_EXPR_STATIC_ASSERT
static_assert(std::is_trivially_destructible_v<TensorData>, "Tensor data must be trivially destructible.");
/////////////// Static assert

TensorData::TensorData(luisa::span<uint32_t const> sizes,
                       TensorElementType element_type,
                       uint64_t uid) noexcept
    : _type(element_type),
      _uid(uid) {
    _size_bytes = 0;
    switch (_type) {
        case TensorElementType::Float16:
            _size_bytes = 2;
            break;
        case TensorElementType::Float32:
            _size_bytes = 4;
            break;
        case TensorElementType::Float64:
            _size_bytes = 8;
            break;
    }
    auto new_sizes = TensorBuilder::get_thd_local()->allocate_array<uint>(sizes.size());
    for (size_t idx = 0; idx != sizes.size(); ++idx) {
        new_sizes[idx] = sizes[idx];
        _size_bytes *= sizes[idx];
    }
    _sizes = luisa::span<uint const>{new_sizes, sizes.size()};
}

TensorData::TensorData(TensorData &&rhs) noexcept = default;
namespace tensor_builder_detail {
static thread_local TensorBuilder *tensor_builder{nullptr};
}// namespace tensor_builder_detail
Tensor::Tensor(TensorData *data,
               bool contained) noexcept
    : _data(data), _contained(contained) {
}
Tensor::Tensor(Tensor &&rhs) noexcept
    : _data(rhs._data),
      _contained(rhs._contained) {
    rhs._contained = false;
}

void Tensor::dispose() noexcept {
    if (_contained) {
        tensor_builder_detail::tensor_builder->deallocate_tensor(_data);
        _contained = false;
    }
    _data = nullptr;
}

Tensor::~Tensor() {
    dispose();
}

void TensorBuilder::deallocate_tensor(TensorData *tensor) noexcept {
    // TODO: record deallocate
}
TensorData *TensorBuilder::allocate_tensor(
    luisa::span<uint32_t const> sizes,
    TensorElementType element_type) noexcept {
    // TODO: record allocate
    auto id = _allocated_tensor.size();
    auto ptr = _tensor_pool.create(
        sizes,
        element_type,
        id);
    _allocated_absolute_size += ptr->size_bytes();
    _allocated_tensor.emplace_back(ptr);
    return ptr;
}
TensorBuilder::TensorBuilder() noexcept {}
TensorBuilder::~TensorBuilder() noexcept {
    for (auto &i : _stack_allocator) {
        luisa::detail::allocator_deallocate(i.ptr, 0);
    }
}
void *TensorBuilder::allocate_stack(size_t size_bytes, size_t alignment) noexcept {
    for (auto &i : _stack_allocator) {
        auto aligned_offset = (i.offset + alignment - 1ull) & (~(alignment - 1ull));
        if (i.size - aligned_offset >= size_bytes) {
            auto result = reinterpret_cast<std::byte *>(i.ptr) + aligned_offset;
            i.offset = aligned_offset + size_bytes;
            return result;
        }
    }
    size_t init_size = 65536ull;
    if (!_stack_allocator.empty()) {
        init_size = _stack_allocator.back().size * 2;
    }
    return _stack_allocator
        .emplace_back(
            Stack{
                .ptr = luisa::detail::allocator_allocate(init_size, 0),
                .size = init_size,
                .offset = size_bytes})
        .ptr;
}

TensorBuilder *TensorBuilder::get_thd_local() {
    return tensor_builder_detail::tensor_builder;
}
void TensorBuilder::set_thd_local(TensorBuilder *builder) {
    tensor_builder_detail::tensor_builder = builder;
}

}// namespace luisa::compute