#include <luisa/tensor/tensor.h>
#include <luisa/tensor/tensor_builder.h>
#include <luisa/core/logging.h>

namespace luisa::compute {
TensorData::TensorData(luisa::span<size_t const> sizes,
                       TensorElementType element_type,
                       uint64_t uid) noexcept
    : _dimension(sizes.size()),
      _type(element_type),
      _uid(uid) {
    size_t _size_bytes = 0;
    switch (_type) {
        case TensorElementType::Float16:
            _size_bytes = 2;
            break;
        case TensorElementType::Float32:
            _size_bytes = 4;
            break;
    }
    if (_dimension > 8) [[unlikely]] {
        LUISA_ERROR("Dimension {} out of range.", _dimension);
    }
    for (size_t idx = 0; idx != sizes.size(); ++idx) {
        _sizes[idx] = sizes[idx];
        _size_bytes *= sizes[idx];
    }
}

TensorData::TensorData(TensorData &&rhs) noexcept
    : _sizes(rhs._sizes),
      _dimension(rhs._dimension),
      _type(rhs._type) {
}

TensorData::~TensorData() noexcept = default;

static thread_local TensorBuilder *tensor_builder{nullptr};

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
        tensor_builder->deallocate_tensor(_data);
        _contained = false;
    }
    _data = nullptr;
}

Tensor::~Tensor() {
    dispose();
}

void TensorBuilder::deallocate_tensor(TensorData* tensor) noexcept {
    // TODO: record deallocate
}
TensorData *TensorBuilder::allocate_tensor(
    luisa::span<size_t const> sizes,
    TensorElementType element_type) noexcept {
    // TODO: record allocate
    auto ptr = _tensor_pool.create(
        sizes,
        element_type,
        _id_counter++);
    _allocated_absolute_size += ptr->size_bytes();
    return ptr;
}
TensorBuilder::TensorBuilder() noexcept {}
TensorBuilder::~TensorBuilder() noexcept {}
}// namespace luisa::compute