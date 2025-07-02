#include <Resource/BindlessArray.h>
#include <Resource/TextureBase.h>
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <DXRuntime/CommandAllocator.h>
#include <luisa/core/logging.h>

namespace lc::dx {

BindlessArray::BindlessArray(
    Device *device, uint arraySize,
    BindlessType type)
    : Resource(device),
      buffer(device, type != BindlessType::None ? 4 : (arraySize * sizeof(BindlessStruct)), device->defaultAllocator.get()) {
    switch (type) {
        case BindlessType::None:
            typed_binded.reset_as(0, arraySize);
            break;
        default: {
            _buffer_node = device->globalHeap->SubAllocate(arraySize);
            typed_binded.reset_as(1, arraySize);
        } break;
    }
}
BindlessArray::~BindlessArray() {
    if (_buffer_node) {
        device->globalHeap->DeAllocate(_buffer_node);
    }
    auto Return = [&](auto &&i) {
        if (i != BindlessStruct::n_pos) {
            device->globalHeap->ReturnIndex(i);
        }
    };
    auto ReturnTex = [&](auto &&i) {
        if (i != BindlessStruct::n_pos) {
            device->globalHeap->ReturnIndex(i & BindlessStruct::mask);
        }
    };
    if (typed_binded.index() == 0) {
        auto &binded = typed_binded.get<0>();
        for (auto &&i : binded) {
            Return(i.first.buffer);
            ReturnTex(i.first.tex2D);
            ReturnTex(i.first.tex3D);
        }
    }
    for (auto &&i : freeQueue) {
        device->globalHeap->ReturnIndex(i);
    }
}
void BindlessArray::Deref(MapIndex &index) {
    if (!index) return;
    auto &&v = index.value();
    v--;
    if (v == 0) {
        ptrMap.remove(index);
    }
    index = {};
}
void BindlessArray::TryReturnIndexTex(MapIndex &index, uint &originValue) {
    if (originValue != BindlessStruct::n_pos) {
        freeQueue.push_back(originValue & BindlessStruct::mask);
        originValue = BindlessStruct::n_pos;
        // device->globalHeap->ReturnIndex(originValue);
        auto &&v = index.value();
        v--;
        if (v == 0) {
            ptrMap.remove(index);
        }
    }
    index = {};
}
void BindlessArray::TryReturnIndex(MapIndex &index, uint &originValue) {
    if (originValue != BindlessStruct::n_pos) {
        freeQueue.push_back(originValue);
        originValue = BindlessStruct::n_pos;
        // device->globalHeap->ReturnIndex(originValue);
        auto &&v = index.value();
        v--;
        if (v == 0) {
            ptrMap.remove(index);
        }
    }
    index = {};
}
BindlessArray::MapIndex BindlessArray::AddIndex(size_t ptr) {
    auto ite = ptrMap.emplace(ptr, 0);
    ite.value()++;
    return ite;
}
void BindlessArray::Bind(vstd::span<const BindlessArrayUpdateCommand::BufferModification> mods) {
    LUISA_DEBUG_ASSERT(typed_binded.index() == 1 && _buffer_node);
    auto &binded = typed_binded.get<1>();
    std::lock_guard lck{mtx};
    if (mods.empty()) return;
    for (auto &&mod : mods) {
        auto &indices = binded[mod.slot];
        Deref(indices);
        using Ope = BindlessArrayUpdateCommand::Modification::Operation;
        if (mod.buffer.op == Ope::EMPLACE) {
            BufferView v{reinterpret_cast<Buffer *>(mod.buffer.handle), mod.buffer.offset_bytes};
            auto newIdx = device->globalHeap->GetSubAllocOffset(_buffer_node) + mod.slot;
            auto desc = v.buffer->GetColorSrvDesc(
                v.offset,
                v.byteSize);
#ifndef NDEBUG
            if (!desc) {
                LUISA_ERROR("illagel buffer");
            }
#endif
            device->globalHeap->CreateSRV(
                v.buffer->GetResource(),
                *desc,
                newIdx);
            indices = AddIndex(mod.buffer.handle);
        }
    }
}

void BindlessArray::Bind(vstd::span<const BindlessArrayUpdateCommand::Texture2DModification> mods) {
    LUISA_DEBUG_ASSERT(typed_binded.index() == 1);
    auto &binded = typed_binded.get<1>();
    std::lock_guard lck{mtx};
    if (mods.empty()) return;
    auto EmplaceTex = [&](uint texIdx, MapIndex &indices, uint64_t handle, TextureBase const *tex, Sampler const &samp) {
        device->globalHeap->CreateSRV(
            tex->GetResource(),
            tex->GetColorSrvDesc(),
            texIdx);
        indices = AddIndex(handle);
    };
    using Ope = BindlessArrayUpdateCommand::Modification::Operation;
    for (auto &&mod : mods) {
        auto vv = mod.slot;
        auto &indices = binded[mod.slot];
        auto newIdx = device->globalHeap->GetSubAllocOffset(_buffer_node) + mod.slot;
        Deref(indices);
        if (mod.tex2d.op == Ope::EMPLACE) {
            EmplaceTex(newIdx, indices, mod.tex2d.handle, reinterpret_cast<TextureBase *>(mod.tex2d.handle), mod.tex2d.sampler);
        }
    }
}

void BindlessArray::Bind(vstd::span<const BindlessArrayUpdateCommand::Modification> mods) {
    LUISA_DEBUG_ASSERT(typed_binded.index() == 0);
    auto &binded = typed_binded.get<0>();
    std::lock_guard lck{mtx};
    if (mods.empty()) return;
    auto EmplaceTex = [&]<bool isTex2D>(BindlessStruct &bindGrp, MapIndicies &indices, uint64_t handle, TextureBase const *tex, Sampler const &samp) {
        if constexpr (isTex2D)
            TryReturnIndexTex(indices.tex2D, bindGrp.tex2D);
        else
            TryReturnIndexTex(indices.tex3D, bindGrp.tex3D);
        auto texIdx = device->globalHeap->AllocateIndex();
        device->globalHeap->CreateSRV(
            tex->GetResource(),
            tex->GetColorSrvDesc(),
            texIdx);
        auto smpIdx = GlobalSamplers::GetIndex(samp);
        if constexpr (isTex2D) {
            indices.tex2D = AddIndex(handle);
            bindGrp.write_samp2d(texIdx, smpIdx);
        } else {
            indices.tex3D = AddIndex(handle);
            bindGrp.write_samp3d(texIdx, smpIdx);
        }
    };
    for (auto &&mod : mods) {
        auto &bindGrp = binded[mod.slot].first;
        auto &indices = binded[mod.slot].second;
        using Ope = BindlessArrayUpdateCommand::Modification::Operation;
        switch (mod.buffer.op) {
            case Ope::REMOVE:
                TryReturnIndex(indices.buffer, bindGrp.buffer);
                break;
            case Ope::EMPLACE: {
                TryReturnIndex(indices.buffer, bindGrp.buffer);
                BufferView v{reinterpret_cast<Buffer *>(mod.buffer.handle), mod.buffer.offset_bytes};
                auto newIdx = device->globalHeap->AllocateIndex();
                auto desc = v.buffer->GetColorSrvDesc(
                    v.offset,
                    v.byteSize);
#ifndef NDEBUG
                if (!desc) {
                    LUISA_ERROR("illagel buffer");
                }
#endif
                device->globalHeap->CreateSRV(
                    v.buffer->GetResource(),
                    *desc,
                    newIdx);
                bindGrp.buffer = newIdx;
                indices.buffer = AddIndex(mod.buffer.handle);
                break;
            }
            default: break;
        }
        switch (mod.tex2d.op) {
            case Ope::REMOVE:
                TryReturnIndexTex(indices.tex2D, bindGrp.tex2D);
                break;
            case Ope::EMPLACE:
                EmplaceTex.operator()<true>(bindGrp, indices, mod.tex2d.handle, reinterpret_cast<TextureBase *>(mod.tex2d.handle), mod.tex2d.sampler);
                break;
            default: break;
        }
        switch (mod.tex3d.op) {
            case Ope::REMOVE:
                TryReturnIndexTex(indices.tex3D, bindGrp.tex3D);
                break;
            case Ope::EMPLACE:
                EmplaceTex.operator()<false>(bindGrp, indices, mod.tex3d.handle, reinterpret_cast<TextureBase *>(mod.tex3d.handle), mod.tex3d.sampler);
                break;
            default: break;
        }
    }
}
void BindlessArray::PreProcessStates(
    CommandBufferBuilder &builder,
    EnhancedBarrierTracker &tracker) const {
    std::lock_guard lck{mtx};
    if (offset_setted && _buffer_node) return;
    tracker.Record(
        BufferView(&buffer),
        _buffer_node ? EnhancedBarrierTracker::Usage::CopyDest : EnhancedBarrierTracker::Usage::ComputeUAV);
}
void BindlessArray::UpdateStates(
    CommandBufferBuilder &builder,
    EnhancedBarrierTracker &tracker,
    vstd::span<const BindlessArrayUpdateCommand::Modification> mods) const {
    LUISA_DEBUG_ASSERT(typed_binded.index() == 0);
    auto &binded = typed_binded.get<0>();
    std::lock_guard lck{mtx};
    struct BindlessElement {
        uint idx;
        BindlessStruct e;
    };
    if (!mods.empty()) {
        auto alloc = builder.GetCB()->GetAlloc();
        auto tempBuffer = alloc->GetTempUploadBuffer(sizeof(BindlessElement) * mods.size(), 16);
        auto ubuffer = static_cast<UploadBuffer const *>(tempBuffer.buffer);
        auto offset = tempBuffer.offset;
        for (auto &&mod : mods) {
            BindlessElement e;
            e.idx = mod.slot;
            e.e = binded[mod.slot].first;
            ubuffer->CopyData(offset, {reinterpret_cast<uint8_t const *>(&e), sizeof(BindlessElement)});
            offset += sizeof(BindlessElement);
        }
        auto cs = device->setBindlessKernel.Get(device);
        auto cbuffer = alloc->GetTempUploadBuffer(sizeof(uint), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
        struct CBuffer {
            uint dsp;
        };
        CBuffer cbValue;
        cbValue.dsp = mods.size();
        static_cast<UploadBuffer const *>(cbuffer.buffer)
            ->CopyData(cbuffer.offset,
                       {reinterpret_cast<uint8_t const *>(&cbValue), sizeof(CBuffer)});
        BindProperty properties[3];
        properties[0] = cbuffer;
        properties[1] = tempBuffer;
        properties[2] = BufferView(&buffer);
        builder.DispatchCompute(
            cs,
            uint3(mods.size(), 1, 1),
            properties);
    }
    if (!freeQueue.empty()) {
        builder.GetCB()->GetAlloc()->ExecuteAfterComplete(
            [vec = std::move(freeQueue),
             device = device] {
                for (auto &&i : vec) {
                    device->globalHeap->ReturnIndex(i);
                }
            });
    }
}
template<typename T>
void BindlessArray::_UpdateStates(
    CommandBufferBuilder &builder,
    EnhancedBarrierTracker &tracker,
    vstd::span<const T> mods) const {
    LUISA_DEBUG_ASSERT(_buffer_node);
    std::lock_guard lck{mtx};
    if (offset_setted) return;
    offset_setted = true;
    auto alloc = builder.GetCB()->GetAlloc();
    auto cbuffer = alloc->GetTempUploadBuffer(sizeof(uint), 16);
    uint value = device->globalHeap->GetSubAllocOffset(_buffer_node);
    static_cast<UploadBuffer const *>(cbuffer.buffer)
        ->CopyData(cbuffer.offset,
                   {reinterpret_cast<uint8_t const *>(&value), sizeof(uint)});
    builder.CopyBuffer(
        cbuffer.buffer,
        &buffer,
        cbuffer.offset,
        0,
        sizeof(uint));
}
void BindlessArray::UpdateStates(
    CommandBufferBuilder &builder,
    EnhancedBarrierTracker &tracker,
    vstd::span<const BindlessArrayUpdateCommand::Texture2DModification> mods) const {
    _UpdateStates<BindlessArrayUpdateCommand::Texture2DModification>(builder, tracker, mods);
}
void BindlessArray::UpdateStates(
    CommandBufferBuilder &builder,
    EnhancedBarrierTracker &tracker,
    vstd::span<const BindlessArrayUpdateCommand::BufferModification> mods) const {
    _UpdateStates<BindlessArrayUpdateCommand::BufferModification>(builder, tracker, mods);
}
}// namespace lc::dx
