#include "dml_ext.h"
#include <luisa/core/dynamic_module.h>
#include <luisa/runtime/stream.h>
#define _D3D12MA_IUNKNOWN_IMPL_FUNCTIONS
#include "DirectMLX.h"
#include <luisa/backends/ext/dx_custom_cmd.h>
#include <wrl/client.h>
#include <Resource/DefaultBuffer.h>

using Microsoft::WRL::ComPtr;
//#include <wil/result_macros.h>
using namespace luisa;
using namespace luisa::compute;
class DMLModule {
    DynamicModule _module;
    std::mutex _mtx;

public:
    DynamicModule &get() {
        std::lock_guard lck{_mtx};
        if (_module) return _module;
        _module = DynamicModule::load("DirectML");
        return _module;
    }
};
static DMLModule g_dml_module;
class DxDMLGraph : public DMLGraph {
public:
    DeviceInterface *device_interface;
    ComPtr<IDMLDevice> dml_device;
    ComPtr<IDMLCompiledOperator> dml_compiled_operator;

    ComPtr<IDMLBindingTable> dml_binding_table;
    ComPtr<IDMLCommandRecorder> dml_command_recorder;
    ComPtr<ID3D12DescriptorHeap> descriptor_heap;
    const size_t weight_size;
    const size_t output_size;
    const size_t input_size;
    size_t desc_count;
    size_t temp_res_count;
    size_t persist_resource_size;

    uint batch_size;
    uint input;
    uint output;
    luisa::vector<uint> hiddens;
    luisa::vector<FusedActivation> activations;

    bool bind = false;
    bool half;
    BufferCreationInfo temporary_buffer{BufferCreationInfo::make_invalid()};
    BufferCreationInfo persistent_buffer{BufferCreationInfo::make_invalid()};
    // ComPtr<ID3D12Resource> temporary_buffer;
    // ComPtr<ID3D12Resource> persistent_buffer;
    unique_ptr<Command> build() noexcept override;
    unique_ptr<Command> forward(Argument::Buffer input_buffer, Argument::Buffer output_buffer, Argument::Buffer weights_buffer) noexcept override;
    DxDMLGraph(
        DeviceInterface *device_interface,
        size_t weight,
        size_t output,
        size_t input,
        size_t batch_size,
        size_t data_size,
        luisa::span<uint const> hiddens,
        luisa::span<const FusedActivation> activations,
        bool half)
        : device_interface(device_interface),
          weight_size(weight * data_size),
          output_size(output * data_size * batch_size),
          input_size(input * data_size * batch_size),
          desc_count(0),
          temp_res_count(0),
          persist_resource_size(0),
          batch_size(batch_size),
          input(input),
          output(output),
          half(half) {
        if (!hiddens.empty()) {
            vstd::push_back_all(this->hiddens, hiddens);
        }
        vstd::push_back_all(this->activations, activations);
    }
    ~DxDMLGraph() override {
        if (temporary_buffer.valid()) {
            device_interface->destroy_buffer(temporary_buffer.handle);
        }
        if (persistent_buffer.valid()) {
            device_interface->destroy_buffer(persistent_buffer.handle);
        }
    }
    [[nodiscard]] size_t input_buffer_size_bytes() const noexcept override {
        return input_size;
    }
    [[nodiscard]] size_t output_buffer_size_bytes() const noexcept override {
        return output_size;
    }
    [[nodiscard]] size_t weight_buffer_size_bytes() const noexcept override {
        return weight_size;
    }
};
class DxGraphBuildCommand final : public DXCustomCmd {
public:
    explicit DxGraphBuildCommand(DxDMLGraph *graph) : _dml_graph(graph) {}
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COMPUTE)

private:
    DxDMLGraph *_dml_graph;
    void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) const noexcept override;
};

static dml::FusedActivation to_dml_activation(FusedActivation a) {
    dml::FusedActivation r;
    r.param1 = a.param1;
    r.param2 = a.param2;
    r.activation = [&]() {
        switch (a.type) {
            case FusedActivation::Type::ELU: return DML_OPERATOR_ACTIVATION_ELU;
            case FusedActivation::Type::HARD_SIGMOID: return DML_OPERATOR_ACTIVATION_HARD_SIGMOID;
            case FusedActivation::Type::IDENTITY: return DML_OPERATOR_ACTIVATION_IDENTITY;
            case FusedActivation::Type::LEAKY_RELU: return DML_OPERATOR_ACTIVATION_LEAKY_RELU;
            case FusedActivation::Type::LINEAR: return DML_OPERATOR_ACTIVATION_LINEAR;
            case FusedActivation::Type::PARAMETRIC_SOFTPLUS: return DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS;
            case FusedActivation::Type::RELU: return DML_OPERATOR_ACTIVATION_RELU;
            case FusedActivation::Type::SCALED_ELU: return DML_OPERATOR_ACTIVATION_SCALED_ELU;
            case FusedActivation::Type::SCALED_TANH: return DML_OPERATOR_ACTIVATION_SCALED_TANH;
            case FusedActivation::Type::SIGMOID: return DML_OPERATOR_ACTIVATION_SIGMOID;
            case FusedActivation::Type::SOFTPLUS: return DML_OPERATOR_ACTIVATION_SOFTPLUS;
            case FusedActivation::Type::SOFTSIGN: return DML_OPERATOR_ACTIVATION_SOFTSIGN;
            case FusedActivation::Type::TANH: return DML_OPERATOR_ACTIVATION_TANH;
            case FusedActivation::Type::THRESHOLDED_RELU: return DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU;
            case FusedActivation::Type::SHRINK: return DML_OPERATOR_ACTIVATION_SHRINK;
            case FusedActivation::Type::CELU: return DML_OPERATOR_ACTIVATION_CELU;
            default: return DML_OPERATOR_INVALID;
        }
    }();
    return r;
}

void DxGraphBuildCommand::execute(IDXGIAdapter1 *adapter, IDXGIFactory2 *dxgi_factory, ID3D12Device *device, ID3D12GraphicsCommandList4 *command_list) const noexcept {
    if (_dml_graph->dml_device) [[unlikely]] {
        LUISA_ERROR("DML Graph already been built.");
    }
    const uint input = _dml_graph->input;
    const uint output = _dml_graph->output;
    const uint batch_size = _dml_graph->batch_size;
    DML_TENSOR_DATA_TYPE dataType = _dml_graph->half ? DML_TENSOR_DATA_TYPE_FLOAT16 : DML_TENSOR_DATA_TYPE_FLOAT32;
    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
    auto &md = g_dml_module.get();
    HRESULT(WINAPI * DMLCreateDevice)
    (
        ID3D12Device * d3d12Device,
        DML_CREATE_DEVICE_FLAGS flags,
        REFIID riid,// Expected: IDMLDevice
        _COM_Outptr_opt_ void **ppv);
    DMLCreateDevice = md.function<std::remove_pointer_t<decltype(DMLCreateDevice)>>("DMLCreateDevice");

    ThrowIfFailed(DMLCreateDevice(
        device,
        dmlCreateDeviceFlags,
        IID_PPV_ARGS(_dml_graph->dml_device.GetAddressOf())));

    dml::Graph graph(_dml_graph->dml_device.Get());
    UINT tensorSizes[4] = {1, 1, UINT(batch_size), UINT(input)};
    dml::TensorDesc::Dimensions inputDimensions(std::begin(tensorSizes), std::end(tensorSizes));
    dml::TensorDesc desc = {dataType, inputDimensions};
    dml::Expression inputLayer = dml::InputTensor(graph, 0, desc);

    vstd::vector<dml::Expression> expressions{};
    expressions.reserve((_dml_graph->hiddens.size() + 1) * 2);
    uint lastDim = input;
    auto &lastOutput = inputLayer;
    for (uint i = 0; i < _dml_graph->hiddens.size(); i++) {
        auto hidden_dim = _dml_graph->hiddens[i];
        UINT matrixSizes[4] = {1, 1, hidden_dim, UINT(lastDim)};
        dml::TensorDesc::Dimensions matrixDimensions = dml::TensorDesc::Dimensions(std::begin(matrixSizes), std::end(matrixSizes));
        auto mdesc = dml::TensorDesc{dataType, matrixDimensions};
        dml::Expression &weights = expressions.emplace_back(dml::InputTensor(graph, i + 1, mdesc));
        dml::Expression &fc = expressions.emplace_back(
            dml::Gemm(lastOutput, weights,
                      dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE, 1.f, 1.f, to_dml_activation(_dml_graph->activations[i])));
        lastDim = hidden_dim;
        lastOutput = fc;
    }
    {
        UINT matrixSizes[4] = {1, 1, UINT(output), UINT(lastDim)};
        dml::TensorDesc::Dimensions matrixDimensions = dml::TensorDesc::Dimensions(std::begin(matrixSizes), std::end(matrixSizes));
        auto mdesc = dml::TensorDesc{dataType, matrixDimensions};
        dml::Expression &weights = expressions.emplace_back(dml::InputTensor(graph, _dml_graph->hiddens.size() + 1, mdesc));
        dml::Expression &fc = expressions.emplace_back(dml::Gemm(lastOutput, weights, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE, 1.f, 1.f, to_dml_activation(_dml_graph->activations.back())));
        lastOutput = fc;
    }

    DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    _dml_graph->dml_compiled_operator.Attach(graph.Compile(executionFlags, {lastOutput}).Detach());

    ComPtr<IDMLOperatorInitializer> dmlOperatorInitializer;
    IDMLCompiledOperator *dml_compiled_operators[] = {_dml_graph->dml_compiled_operator.Get()};
    ThrowIfFailed(_dml_graph->dml_device->CreateOperatorInitializer(
        vstd::array_count(dml_compiled_operators),
        dml_compiled_operators,
        IID_PPV_ARGS(dmlOperatorInitializer.GetAddressOf())));

    // Query the operator for the required size (in descriptors) of its binding table.
    // You need to initialize an operator exactly once before it can be executed, and
    // the two stages require different numbers of descriptors for binding. For simplicity,
    // we create a single descriptor heap that's large enough to satisfy them both.
    DML_BINDING_PROPERTIES initializeBindingProperties = dmlOperatorInitializer->GetBindingProperties();
    DML_BINDING_PROPERTIES executeBindingProperties = _dml_graph->dml_compiled_operator->GetBindingProperties();
    _dml_graph->desc_count = std::max(
        initializeBindingProperties.RequiredDescriptorCount,
        executeBindingProperties.RequiredDescriptorCount);

    // Create descriptor heaps.

    D3D12_DESCRIPTOR_HEAP_DESC descriptor_heap_desc{};
    descriptor_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptor_heap_desc.NumDescriptors = _dml_graph->desc_count;
    descriptor_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(device->CreateDescriptorHeap(
        &descriptor_heap_desc,
        IID_PPV_ARGS(_dml_graph->descriptor_heap.GetAddressOf())));

    // Set the descriptor heap(s).
    ID3D12DescriptorHeap *d3D12DescriptorHeaps[] = {_dml_graph->descriptor_heap.Get()};
    command_list->SetDescriptorHeaps(vstd::array_count(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    // Create a binding table over the descriptor heap we just created.
    DML_BINDING_TABLE_DESC dml_binding_table_desc{};
    dml_binding_table_desc.Dispatchable = dmlOperatorInitializer.Get();
    dml_binding_table_desc.CPUDescriptorHandle = _dml_graph->descriptor_heap->GetCPUDescriptorHandleForHeapStart();
    dml_binding_table_desc.GPUDescriptorHandle = _dml_graph->descriptor_heap->GetGPUDescriptorHandleForHeapStart();
    dml_binding_table_desc.SizeInDescriptors = _dml_graph->desc_count;

    ComPtr<IDMLBindingTable> initBindingTable;
    ThrowIfFailed(_dml_graph->dml_device->CreateBindingTable(
        &dml_binding_table_desc,
        IID_PPV_ARGS(initBindingTable.GetAddressOf())));

    // Create the temporary and persistent resources that are necessary for executing an operator.

    // The temporary resource is scratch memory (used internally by DirectML), whose contents you don't need to define.
    // The persistent resource is long-lived, and you need to initialize it using the IDMLOperatorInitializer.

    _dml_graph->temp_res_count = std::max(
        initializeBindingProperties.TemporaryResourceSize,
        executeBindingProperties.TemporaryResourceSize);
    _dml_graph->persist_resource_size = executeBindingProperties.PersistentResourceSize;

    // Bind and initialize the operator on the GPU.

    if (_dml_graph->temp_res_count != 0) {
        _dml_graph->temporary_buffer = _dml_graph->device_interface->create_buffer(
            Type::of<void>() /*nullptr*/,
            _dml_graph->temp_res_count,
            nullptr);
        if (initializeBindingProperties.TemporaryResourceSize != 0) {
            DML_BUFFER_BINDING bufferBinding{
                reinterpret_cast<ID3D12Resource *>(_dml_graph->temporary_buffer.native_handle),
                0, _dml_graph->temp_res_count};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            initBindingTable->BindTemporaryResource(&bindingDesc);
        }
    }

    if (_dml_graph->persist_resource_size != 0) {
        _dml_graph->persistent_buffer = _dml_graph->device_interface->create_buffer(
            Type::of<void>() /*nullptr*/,
            _dml_graph->persist_resource_size,
            nullptr);
        // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
        DML_BUFFER_BINDING bufferBinding{
            reinterpret_cast<ID3D12Resource *>(_dml_graph->persistent_buffer.native_handle),
            0, _dml_graph->persist_resource_size};
        DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
        initBindingTable->BindOutputs(1, &bindingDesc);
    }

    // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
    ThrowIfFailed(_dml_graph->dml_device->CreateCommandRecorder(
        IID_PPV_ARGS(_dml_graph->dml_command_recorder.GetAddressOf())));

    _dml_graph->dml_command_recorder->RecordDispatch(
        command_list,
        dmlOperatorInitializer.Get(),
        initBindingTable.Get());

    ThrowIfFailed(_dml_graph->dml_device->CreateBindingTable(
        &dml_binding_table_desc,
        IID_PPV_ARGS(_dml_graph->dml_binding_table.GetAddressOf())));
}

class DxGraphForwardCommand final : public DXCustomCmd {
    luisa::vector<EnhancedResourceUsage> _resource_usages;
    luisa::span<EnhancedResourceUsage> get_enhanced_resource_usages() noexcept override {
        return _resource_usages;
    }
public:
    DxGraphForwardCommand(DxDMLGraph *graph, Argument::Buffer const &ipt, Argument::Buffer const &opt, Argument::Buffer const &w)
        : _dml_graph(graph),
          _input(static_cast<lc::dx::DefaultBuffer *>(reinterpret_cast<lc::dx::Buffer *>(ipt.handle))->GetResource()),
          _output(static_cast<lc::dx::DefaultBuffer *>(reinterpret_cast<lc::dx::Buffer *>(opt.handle))->GetResource()),
          _weight(static_cast<lc::dx::DefaultBuffer *>(reinterpret_cast<lc::dx::Buffer *>(w.handle))->GetResource()) {
        if (ipt.size != graph->input_size) [[unlikely]] {
            LUISA_ERROR("Input buffer size {} mismatch. required {}", ipt.size, graph->input_size);
        }
        if (opt.size != graph->output_size) [[unlikely]] {
            LUISA_ERROR("Output buffer size {} mismatch. required {}", opt.size, graph->output_size);
        }
        if (w.size != graph->weight_size) [[unlikely]] {
            LUISA_ERROR("Weight buffer size {} mismatch. required {}", w.size, graph->weight_size);
        }
        _resource_usages.emplace_back(
            ipt,
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);
        _resource_usages.emplace_back(
            opt,
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);
        _resource_usages.emplace_back(
            w,
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);
        if (_dml_graph->temporary_buffer.valid()) {
            _resource_usages.emplace_back(
                Argument::Buffer{
                    .handle = _dml_graph->temporary_buffer.handle,
                    .offset = 0,
                    .size = _dml_graph->temporary_buffer.total_size_bytes},
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);
        }
        if (_dml_graph->persistent_buffer.valid()) {
            _resource_usages.emplace_back(
                Argument::Buffer{
                    .handle = _dml_graph->persistent_buffer.handle,
                    .offset = 0,
                    .size = _dml_graph->persistent_buffer.total_size_bytes},
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);
        }
    }
    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }

private:
    DxDMLGraph *_dml_graph;
    ID3D12Resource *_input;
    ID3D12Resource *_output;
    ID3D12Resource *_weight;

    void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) const noexcept override;
};

void DxGraphForwardCommand::execute(IDXGIAdapter1 *adapter, IDXGIFactory2 *dxgi_factory, ID3D12Device *device, ID3D12GraphicsCommandList4 *command_list) const noexcept {
    // const uint layer = _dml_graph->hiddens.size() + 1;

    uint data_size = _dml_graph->half ? 2 : 4;
    if (!_dml_graph->bind) {
        _dml_graph->bind = true;
        DML_BINDING_TABLE_DESC dml_binding_table_desc{};
        dml_binding_table_desc.Dispatchable = _dml_graph->dml_compiled_operator.Get();
        dml_binding_table_desc.CPUDescriptorHandle = _dml_graph->descriptor_heap->GetCPUDescriptorHandleForHeapStart();
        dml_binding_table_desc.GPUDescriptorHandle = _dml_graph->descriptor_heap->GetGPUDescriptorHandleForHeapStart();
        dml_binding_table_desc.SizeInDescriptors = _dml_graph->desc_count;
        dml_binding_table_desc.Dispatchable = _dml_graph->dml_compiled_operator.Get();
        ThrowIfFailed(_dml_graph->dml_binding_table->Reset(&dml_binding_table_desc));

        if (_dml_graph->temp_res_count != 0) {
            DML_BUFFER_BINDING bufferBinding{reinterpret_cast<ID3D12Resource *>(_dml_graph->temporary_buffer.native_handle), 0, _dml_graph->temp_res_count};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            _dml_graph->dml_binding_table->BindTemporaryResource(&bindingDesc);
        }
        if (_dml_graph->persist_resource_size != 0) {
            DML_BUFFER_BINDING bufferBinding{reinterpret_cast<ID3D12Resource *>(_dml_graph->persistent_buffer.native_handle), 0, _dml_graph->persist_resource_size};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            _dml_graph->dml_binding_table->BindPersistentResource(&bindingDesc);
        }
        {

            vstd::vector<DML_BINDING_DESC> inputBindingDescs{};
            vstd::vector<DML_BUFFER_BINDING> inputBufferBindings{};
            inputBufferBindings.resize(_dml_graph->hiddens.size() + 2);
            inputBufferBindings[0] = DML_BUFFER_BINDING{_input, 0, _dml_graph->input_size};
            inputBindingDescs.emplace_back(DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &inputBufferBindings[0]});
            uint lastDim = _dml_graph->input;
            size_t offset = 0;
            for (uint i = 0; i < _dml_graph->hiddens.size(); i++) {
                auto hidden_dim = _dml_graph->hiddens[i];
                inputBufferBindings[i + 1] = DML_BUFFER_BINDING{_weight, offset, size_t(lastDim) * hidden_dim * data_size};
                inputBindingDescs.emplace_back(DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &inputBufferBindings[i + 1]});
                offset += inputBufferBindings[i + 1].SizeInBytes;
                lastDim = hidden_dim;
            }
            {
                inputBufferBindings[_dml_graph->hiddens.size() + 1] = DML_BUFFER_BINDING{_weight, offset, size_t(lastDim) * _dml_graph->output * data_size};
                inputBindingDescs.emplace_back(DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &inputBufferBindings[_dml_graph->hiddens.size() + 1]});
            }

            _dml_graph->dml_binding_table->BindInputs(inputBindingDescs.size(), inputBindingDescs.data());
        }
        {
            DML_BUFFER_BINDING outputBufferBinding{_output, 0, _dml_graph->output_size};
            DML_BINDING_DESC outputBindingDesc{DML_BINDING_TYPE_BUFFER, &outputBufferBinding};
            _dml_graph->dml_binding_table->BindOutputs(1, &outputBindingDesc);
        }
    }
    ID3D12DescriptorHeap *d3D12DescriptorHeaps[] = {_dml_graph->descriptor_heap.Get()};
    command_list->SetDescriptorHeaps(vstd::array_count(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    //Dispatch the operator
    _dml_graph->dml_command_recorder->RecordDispatch(command_list, _dml_graph->dml_compiled_operator.Get(), _dml_graph->dml_binding_table.Get());
}

lc::dx::DxDirectMLExt::DxDirectMLExt(DeviceInterface *device) : device(device) {
}

luisa::unique_ptr<DMLGraph> lc::dx::DxDirectMLExt::create_graph(
    uint32_t batch_size,
    uint32_t input_elements,
    uint32_t out_elements,
    luisa::span<const uint32_t> hidden_layer_elements,
    luisa::span<const FusedActivation> activations,
    bool half_precision) noexcept {
    uint data_size = half_precision ? 2 : 4;
    size_t weight_size = 0;
    auto last_size = input_elements;
    for (auto &&i : hidden_layer_elements) {
        weight_size += last_size * i;
        last_size = i;
    }
    weight_size += last_size * out_elements;
    if (activations.size() != (hidden_layer_elements.size() + 1)) [[unlikely]] {
        LUISA_ERROR("Hidden layers' and activations' size mismatch.");
    }
    auto graph = luisa::make_unique<DxDMLGraph>(
        device,
        weight_size,
        out_elements,
        input_elements,
        batch_size,
        data_size,
        hidden_layer_elements,
        activations,
        half_precision);
    return graph;
}

unique_ptr<Command> DxDMLGraph::build() noexcept {
    return luisa::make_unique<DxGraphBuildCommand>(this);
}
unique_ptr<Command> DxDMLGraph::forward(Argument::Buffer input_buffer, Argument::Buffer output_buffer, Argument::Buffer weights_buffer) noexcept {
    return luisa::make_unique<DxGraphForwardCommand>(this, input_buffer, output_buffer, weights_buffer);
}
