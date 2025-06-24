#include "stream.h"
#include "device.h"
#include "compute_shader.h"
#include "bindless_array.h"
#include <luisa/core/logging.h>
#include "log.h"
#include "blas.h"
#include "tlas.h"
namespace lc::vk {
template<typename Visitor>
void DecodeCmd(vstd::span<const Argument> args, Visitor &&visitor) {
    using Tag = Argument::Tag;
    for (auto &&i : args) {
        switch (i.tag) {
            case Tag::BUFFER: {
                visitor(i.buffer);
            } break;
            case Tag::TEXTURE: {
                visitor(i.texture);
            } break;
            case Tag::UNIFORM: {
                visitor(i.uniform);
            } break;
            case Tag::BINDLESS_ARRAY: {
                visitor(i.bindless_array);
            } break;
            case Tag::ACCEL: {
                visitor(i.accel);
            } break;
            default: {
                LUISA_ASSUME(false);
            } break;
        }
    }
}
bool ReorderFuncTable::is_res_in_bindless(uint64_t bindless_handle, uint64_t resource_handle) const noexcept {
    return reinterpret_cast<BindlessArray *>(bindless_handle)->is_ptr_in_bindless(resource_handle);
}
void ReorderFuncTable::lock_bindless(uint64_t bindless_handle) const noexcept {
    reinterpret_cast<BindlessArray *>(bindless_handle)->mtx.lock();
}
void ReorderFuncTable::unlock_bindless(uint64_t bindless_handle) const noexcept {
    reinterpret_cast<BindlessArray *>(bindless_handle)->mtx.unlock();
}
void ReorderFuncTable::update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Modification> modifications) const noexcept {
    reinterpret_cast<BindlessArray *>(handle)->bind(modifications);
}
struct ResourceBarrierVisitor {
    ResourceBarrier *barrier;
    SavedArgument const *arg;
    vstd::vector<std::byte> *arg_buffer;
    ShaderDispatchCommandBase const &cmd;
    ResourceBarrier::Usage uav_usage;
    ResourceBarrier::Usage read_usage;
    ResourceBarrier::Usage accel_read_usage;
    template<typename T>
    void emplace_data(T const &data) {
        size_t sz = arg_buffer->size();
        luisa::enlarge_by(*arg_buffer, sizeof(T));
        using PlaceHolder = luisa::aligned_storage_t<sizeof(T), 1>;
        *reinterpret_cast<PlaceHolder *>(arg_buffer->data() + sz) =
            *reinterpret_cast<PlaceHolder const *>(&data);
    }
    template<typename T>
    void emplace_data(T const *data, size_t size) {
        size_t sz = arg_buffer->size();
        auto byteSize = size * sizeof(T);
        luisa::enlarge_by(*arg_buffer, byteSize);
        std::memcpy(arg_buffer->data() + sz, data, byteSize);
    }
    ResourceBarrierVisitor(
        ResourceBarrier *barrier,
        SavedArgument const *arg,
        vstd::vector<std::byte> *arg_buffer,
        ShaderDispatchCommandBase const &cmd,
        bool is_raster) : barrier(barrier), arg(arg), arg_buffer(arg_buffer), cmd(cmd) {
        if (is_raster) {
            uav_usage = ResourceBarrier::Usage::RasterUAV;
            read_usage = ResourceBarrier::Usage::RasterRead;
            accel_read_usage = ResourceBarrier::Usage::RasterAccelRead;
        } else {
            uav_usage = ResourceBarrier::Usage::ComputeUAV;
            read_usage = ResourceBarrier::Usage::ComputeRead;
            accel_read_usage = ResourceBarrier::Usage::ComputeAccelRead;
        }
    }
    void operator()(Argument::Buffer const &bf) {
        auto res = reinterpret_cast<Buffer const *>(bf.handle);
        if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
            // LUISA_ASSERT(is_device_buffer(res), "Unordered access buffer can not be host-buffer.");
            barrier->record(
                BufferView{res, bf.offset, bf.size},
                uav_usage);
        } else {
            barrier->record(
                BufferView{res, bf.offset, bf.size},
                read_usage);
        }
        ++arg;
    }
    void operator()(Argument::Texture const &bf) {
        auto rt = reinterpret_cast<Texture *>(bf.handle);
        //UAV
        if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
            barrier->record(
                TexView{rt, bf.level},
                uav_usage);
        }
        // SRV
        else {
            barrier->record(
                TexView{rt, bf.level},
                read_usage);
        }
        ++arg;
    }
    void operator()(Argument::BindlessArray const &bf) {
        auto bdls = reinterpret_cast<BindlessArray *>(bf.handle);
        auto &buffer = bdls->indices_buffer();
        barrier->record(
            BufferView(&buffer, 0, buffer.byte_size()),
            read_usage);
        barrier->process_bindless(bdls, read_usage);
        ++arg;
    }
    void operator()(Argument::Uniform const &a) {
        auto bf = cmd.uniform(a);
        if (bf.size() < 4) {
            bool v = (bool)bf[0];
            uint value = v ? std::numeric_limits<uint>::max() : 0;
            emplace_data(value);
        } else {
            emplace_data(bf.data(), arg->structSize);
        }
        ++arg;
    }
    void operator()(Argument::Accel const &bf) {
        auto tlas = reinterpret_cast<Tlas *>(bf.handle);
        if ((luisa::to_underlying(arg->varUsage) & luisa::to_underlying(Usage::WRITE)) != 0) {
            barrier->record(
                BufferView(tlas->instance_buffer()),
                ResourceBarrier::Usage::ComputeUAV);
        } else {
            barrier->record(
                BufferView(tlas->instance_buffer()),
                ResourceBarrier::Usage::ComputeRead);
            barrier->record(
                BufferView(tlas->accel_buffer()),
                ResourceBarrier::Usage::ComputeAccelRead);
        }
        ++arg;
    }
};
struct BindPropVisitor {
    // Each sets
    CommandBuffer *cmdbuffer;
    VkDescriptorSet desc_set;
    uint desc_index;
    vstd::vector<VkImageView> *img_views;
    SavedArgument const *arg;
    void operator()(Argument::Buffer const &bf) {
        auto idx = desc_index++;
        auto buffer_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
        *buffer_descs = VkDescriptorBufferInfo{
            reinterpret_cast<DefaultBuffer const *>(bf.handle)->vk_buffer(),
            bf.offset,
            bf.size};
        auto &a = cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            desc_set,
            idx,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            buffer_descs,
            nullptr});
        ++arg;
    }
    void operator()(Argument::Texture const &bf) {
        auto idx = desc_index++;
        auto tex = reinterpret_cast<Texture const *>(bf.handle);

        VkImageViewCreateInfo imgview_create_info{
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            nullptr,
            0,
            tex->vk_image(),
            VkImageViewType(tex->dimension() - 1),
            Texture::to_vk_format(tex->format()),
            VkComponentMapping{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
            VkImageSubresourceRange{
                VK_IMAGE_ASPECT_COLOR_BIT,
                bf.level,
                1,
                0,
                1}};
        VkImageView img_view;
        VK_CHECK_RESULT(vkCreateImageView(cmdbuffer->device()->logic_device(), &imgview_create_info, Device::alloc_callbacks(), &img_view));
        img_views->emplace_back(img_view);
        auto image_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorImageInfo>();
        *image_descs = VkDescriptorImageInfo{
            VkSampler{nullptr},
            img_view,
            cmdbuffer->resource_barrier->get_layout(tex, bf.level)};

        cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            desc_set,
            idx,
            0,
            1,
            ((luisa::to_underlying(arg->varUsage) & luisa::to_underlying(Usage::WRITE)) != 0) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            image_descs,
            nullptr,
            nullptr});
        ++arg;
    }
    void operator()(Argument::Uniform const &a) {
    }
    void operator()(Argument::BindlessArray const &bf) {
        auto &buffer = reinterpret_cast<BindlessArray const *>(bf.handle)->indices_buffer();
        auto idx = desc_index++;
        auto buffer_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
        *buffer_descs = VkDescriptorBufferInfo{
            buffer.vk_buffer(),
            0,
            buffer.byte_size()};
        auto &a = cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            desc_set,
            idx,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            buffer_descs,
            nullptr});
        ++arg;
    }
    void operator()(Argument::Accel const &bf) {
        auto tlas = reinterpret_cast<Tlas *>(bf.handle);
        if ((luisa::to_underlying(arg->varUsage) & luisa::to_underlying(Usage::WRITE)) != 0) {
            auto idx = desc_index++;
            auto buffer_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
            *buffer_descs = VkDescriptorBufferInfo{
                tlas->instance_buffer()->vk_buffer(),
                0,
                tlas->instance_buffer()->byte_size()};
            auto &a = cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                desc_set,
                idx,
                0,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                buffer_descs,
                nullptr});
        } else {
            // accel
            {
                auto idx = desc_index++;
                auto accel_info = cmdbuffer->temp_desc->allocate_memory<VkWriteDescriptorSetAccelerationStructureKHR>();
                accel_info->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
                accel_info->accelerationStructureCount = 1;
                accel_info->pAccelerationStructures = &tlas->accel();
                auto &a = cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    accel_info,
                    desc_set,
                    idx,
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                    nullptr,
                    nullptr,
                    nullptr});
            }
            // instance
            {
                auto idx = desc_index++;
                auto buffer_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
                *buffer_descs = VkDescriptorBufferInfo{
                    tlas->instance_buffer()->vk_buffer(),
                    0,
                    tlas->instance_buffer()->byte_size()};
                auto &a = cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    desc_set,
                    idx,
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    buffer_descs,
                    nullptr});
            }
        }
        ++arg;
    }
};
namespace temp_buffer {
uint64 DefaultBufferDeferredVisitor::allocate(uint64 size) {
    auto bf = new DefaultBuffer(device, size);
    _buffers.try_emplace(
        reinterpret_cast<uint64_t>(bf),
        bf);
    return reinterpret_cast<uint64>(bf);
}
void DefaultBufferDeferredVisitor::deallocate(uint64 handle) {
    if (_buffers.empty()) return;
    auto iter = _buffers.find(handle);
    LUISA_ASSERT(iter != _buffers.end());
    cmdbuffer->states()->dispose_after_flush(std::move(iter->second));
    _buffers.erase(iter);
}
template<typename Pack>
uint64 Visitor<Pack>::allocate(uint64 size) {
    return reinterpret_cast<uint64_t>(new Pack(device, size));
}
template<typename Pack>
void Visitor<Pack>::deallocate(uint64 handle) {
    delete reinterpret_cast<Pack *>(handle);
}
template<typename Pack>
auto Visitor<Pack>::Create(uint64 size) -> Pack * {
    return new Pack{device, size};
}
template<typename T>
void BufferAllocator<T>::clear() {
    largeBuffers.clear();
    alloc.dispose();
}
template<typename T>
BufferAllocator<T>::BufferAllocator(size_t initCapacity)
    : alloc(initCapacity, &visitor) {
}
template<typename T>
BufferAllocator<T>::~BufferAllocator() {
}
template<typename T>
BufferView BufferAllocator<T>::allocate(size_t size) {
    if (size <= kLargeBufferSize) [[likely]] {
        auto chunk = alloc.allocate(size);
        return BufferView(reinterpret_cast<T const *>(chunk.handle), chunk.offset, size);
    } else {
        auto &v = largeBuffers.emplace_back(visitor.Create(size));
        return BufferView(v.get(), 0, size);
    }
}

template<typename T>
BufferView BufferAllocator<T>::allocate(size_t size, size_t align) {
    if (size <= kLargeBufferSize) [[likely]] {
        auto chunk = alloc.allocate(size, align);
        return BufferView(reinterpret_cast<T const *>(chunk.handle), chunk.offset, size);
    } else {
        auto &v = largeBuffers.emplace_back(visitor.Create(size));
        return BufferView(v.get(), 0, size);
    }
}
}// namespace temp_buffer

static size_t TEMP_SIZE = 1024ull * 1024ull;
CommandBufferState::CommandBufferState()
    : upload_alloc(TEMP_SIZE),
      readback_alloc(TEMP_SIZE) {
}
void CommandBufferState::init(Device &device, StreamTag tag) {
    this->device = &device;
    upload_alloc.visitor.device = &device;
    readback_alloc.visitor.device = &device;
    {
        VkDescriptorPoolSize pool_sizes[3];
        pool_sizes[0].descriptorCount = 262144;
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[1].descriptorCount = 262144;
        pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        pool_sizes[2].descriptorCount = 262144;
        pool_sizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        VkDescriptorPoolCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 65536,
            .poolSizeCount = vstd::array_count(pool_sizes),
            .pPoolSizes = pool_sizes};
        VK_CHECK_RESULT(vkCreateDescriptorPool(device.logic_device(), &createInfo, Device::alloc_callbacks(), &_desc_pool));
    }
    if (!_pool) {
        VkCommandPoolCreateInfo pool_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
        switch (tag) {
            case StreamTag::GRAPHICS:
                pool_ci.queueFamilyIndex = device.graphics_queue_index();
                break;
            case StreamTag::COPY:
                pool_ci.queueFamilyIndex = device.copy_queue_index();
                break;
            case StreamTag::COMPUTE:
                pool_ci.queueFamilyIndex = device.compute_queue_index();
                break;
            default:
                LUISA_ASSERT(false, "Illegal stream tag.");
        }
        VK_CHECK_RESULT(vkCreateCommandPool(device.logic_device(), &pool_ci, Device::alloc_callbacks(), &_pool));
    }
}
CommandBufferState::~CommandBufferState() {
    vkDestroyCommandPool(device->logic_device(), _pool, Device::alloc_callbacks());
    vkDestroyDescriptorPool(device->logic_device(), _desc_pool, Device::alloc_callbacks());
}
void CommandBufferState::reset(Device &device) {
    for (auto &i : _callbacks) {
        i();
    }
    _callbacks.clear();
    for (auto &i : _dispose_pool) {
        i.second(i.first);
    }
    _dispose_pool.clear();
    upload_alloc.clear();
    readback_alloc.clear();
    for (auto i : img_views) {
        vkDestroyImageView(device.logic_device(), i, Device::alloc_callbacks());
    }
    img_views.clear();
    VK_CHECK_RESULT(vkResetDescriptorPool(device.logic_device(), _desc_pool, 0));
}
void CommandBuffer::reset() {
    VK_CHECK_RESULT(vkResetCommandBuffer(_cmdbuffer, 0));
    _state->reset(*device());
}

Stream::Stream(Device *device, StreamTag tag)
    : Resource{device},
      _evt(device),
      reorder({}),
      _thd([this]() {
          auto loop_cmd = [&]() {
              while (auto p = _exec.pop()) {
                  p->visit(
                      [&]<typename T>(T &t) {
                          if constexpr (std::is_same_v<T, Callbacks>) {
                              for (auto &i : t) {
                                  i();
                              }
                          } else if constexpr (std::is_same_v<T, SyncExt>) {
                              t.evt->host_wait(t.value);
                          } else if constexpr (std::is_same_v<T, NotifyEvt>) {
                              t.evt->notify(t.value);
                          } else if constexpr (std::is_same_v<T, CommandBuffer>) {
                              t.reset();
                              _cmdbuffers.push(std::move(t));
                          }
                      });
              }
          };
          while (_enabled) {
              loop_cmd();
              std::unique_lock lck{_mtx};
              while (_enabled && _exec.length() == 0) {
                  _cv.wait(lck);
              }
          }
          loop_cmd();
      }),
      temp_desc(65536, &temp_desc_visitor, 2),
      scratch_buffer_alloc(TEMP_SIZE, &scratch_buffer_alloc_visitor),
      _stream_tag(tag) {
    switch (tag) {
        case StreamTag::GRAPHICS:
            _queue = device->graphics_queue();
            resource_barrier.queue_type = ResourceBarrier::QueueType::Graphics;
            resource_barrier.queue_index = device->graphics_queue_index();
            break;
        case StreamTag::COPY:
            resource_barrier.queue_type = ResourceBarrier::QueueType::Copy;
            _queue = device->copy_queue();
            resource_barrier.queue_index = device->copy_queue_index();
            break;
        case StreamTag::COMPUTE:
            resource_barrier.queue_type = ResourceBarrier::QueueType::Compute;
            _queue = device->compute_queue();
            resource_barrier.queue_index = device->compute_queue_index();
            break;
        default:
            LUISA_ASSERT(false, "Illegal stream tag.");
    }
}
Stream::~Stream() {
    sync();
    {
        std::lock_guard lck{_mtx};
        _enabled = false;
    }
    _cv.notify_one();
    _thd.join();
    scratch_buffer_alloc_visitor._buffers.clear();
    while (auto p = _cmdbuffers.pop()) {
    }
}
void Stream::dispatch(
    vstd::span<const luisa::unique_ptr<Command>> cmds,
    luisa::vector<luisa::move_only_function<void()>> &&callbacks,
    bool inqueue_limit) {

    if (cmds.empty() && callbacks.empty()) {
        return;
    }
    if (inqueue_limit) {
        if (_evt.last_fence() > 2) {
            _evt.sync(_evt.last_fence() - 2);
        }
    }
    auto fence = _evt.last_fence() + 1;
    if (!cmds.empty()) {
        CommandBuffer cmdbuffer = [&]() {
            auto p = _cmdbuffers.pop();
            if (p) return std::move(*p);
            return CommandBuffer{*this};
        }();
        scratch_buffer_alloc_visitor.cmdbuffer = &cmdbuffer;
        scratch_buffer_alloc_visitor.device = device();

        auto cb = cmdbuffer.cmdbuffer();

        cmdbuffer.resource_barrier = &resource_barrier;
        cmdbuffer.uniform_data = &uniform_data;
        cmdbuffer.desc_sets = &desc_sets;
        cmdbuffer.dispatch_offsets = &dispatch_offsets;
        cmdbuffer.write_desc_sets = &write_desc_sets;
        cmdbuffer.bindless_cache = &bindless_cache;
        cmdbuffer.temp_desc = &temp_desc;
        cmdbuffer.scratch_buffer_alloc = &scratch_buffer_alloc;
        cmdbuffer.begin();
        cmdbuffer.execute(cmds);
        cmdbuffer.end();
        _evt.signal(*this, fence, &cb);
        _exec.push(SyncExt{
            .evt = &_evt,
            .value = fence});
        _exec.push(std::move(cmdbuffer));
    } else {
        _evt.update_fence(fence);
    }
    if (!callbacks.empty()) {
        _exec.push(std::move(callbacks));
    }
    _exec.push(NotifyEvt{
        .evt = &_evt,
        .value = fence});

    _mtx.lock();
    _mtx.unlock();
    _cv.notify_one();
}
void Stream::sync() {
    _evt.sync(_evt.last_fence());
}
CommandBuffer::CommandBuffer(Stream &stream)
    : Resource(stream.device()),
      stream(stream),
      _state(vstd::make_unique<CommandBufferState>()) {
    _state->init(*stream.device(), stream.stream_tag());
    VkCommandBufferAllocateInfo cb_ci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = _state->_pool,
        .commandBufferCount = 1};
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device()->logic_device(), &cb_ci, &_cmdbuffer));
    // VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    // VK_CHECK_RESULT(vkCreateFence(device()->logic_device(), &fence_info, Device::alloc_callbacks(), nullptr));
}
CommandBuffer::~CommandBuffer() {
    if (_cmdbuffer)
        vkFreeCommandBuffers(device()->logic_device(), _state->_pool, 1, &_cmdbuffer);
}
void CommandBuffer::begin() {
    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    VK_CHECK_RESULT(vkBeginCommandBuffer(_cmdbuffer, &bi));
}
void CommandBuffer::end() {
    VK_CHECK_RESULT(vkEndCommandBuffer(_cmdbuffer));
}
CommandBuffer::CommandBuffer(CommandBuffer &&rhs)
    : Resource(std::move(rhs)),
      stream(rhs.stream),
      _cmdbuffer(rhs._cmdbuffer),
      _state(std::move(rhs._state)) {
    rhs._cmdbuffer = nullptr;
}
void Stream::signal(Event *event, uint64_t value) {
    event->signal(*this, value);
    _exec.push(SyncExt{event, value});
    _exec.push(NotifyEvt{event, value});
    _mtx.lock();
    _mtx.unlock();
    _cv.notify_one();
}
void Stream::wait(Event *event, uint64_t value) {
    event->wait(*this, value);
}
void CommandBuffer::execute(vstd::span<const luisa::unique_ptr<Command>> cmds) {
    for (auto &&command : cmds) {
        command->accept(stream.reorder);
    }
    auto cmd_lists = stream.reorder.command_lists();
    auto clear_reorder = vstd::scope_exit([&] {
        stream.reorder.clear();
    });
    for (auto &&lst : cmd_lists) {
        dispatch_offsets->clear();
        uniform_data->clear();
        auto delay_clear = vstd::scope_exit([&]() {
            scratch_buffer_alloc->clear();
        });
        // Preprocess: record resources' states
        for (auto i = lst; i != nullptr; i = i->p_next) {
            auto cmd = i->cmd;
            switch (cmd->tag()) {
                case Command::Tag::EBufferUploadCommand: {
                    auto c = static_cast<BufferUploadCommand const *>(cmd);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->handle()),
                            c->offset(),
                            c->size()},
                        ResourceBarrier::Usage::CopyDest);
                } break;
                case Command::Tag::EBufferDownloadCommand: {
                    auto c = static_cast<BufferDownloadCommand const *>(cmd);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->handle()),
                            c->offset(),
                            c->size()},
                        ResourceBarrier::Usage::CopySource);
                } break;
                case Command::Tag::EBufferCopyCommand: {
                    auto c = static_cast<BufferCopyCommand const *>(cmd);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->dst_handle()),
                            c->dst_offset(),
                            c->size()},
                        ResourceBarrier::Usage::CopyDest);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->src_handle()),
                            c->src_offset(),
                            c->size()},
                        ResourceBarrier::Usage::CopySource);
                } break;
                case Command::Tag::EBufferToTextureCopyCommand: {
                    auto c = static_cast<BufferToTextureCopyCommand const *>(cmd);
                    resource_barrier->record(
                        TexView{
                            reinterpret_cast<Texture const *>(c->texture()),
                            c->level()},
                        ResourceBarrier::Usage::CopySource);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->buffer()),
                            c->buffer_offset(),
                            pixel_storage_size(c->storage(), c->size())},
                        ResourceBarrier::Usage::CopyDest);
                } break;
                case Command::Tag::EShaderDispatchCommand: {
                    auto c = static_cast<ShaderDispatchCommand const *>(cmd);
                    auto shader = reinterpret_cast<Shader const *>(c->handle());
                    uniform_data->resize_uninitialized((uniform_data->size() + 15) & (~(15ull)));
                    std::pair<size_t, size_t> sizes;
                    sizes.first = uniform_data->size_bytes();
                    ResourceBarrierVisitor visitor{
                        resource_barrier,
                        shader->saved_arguments().data(),
                        uniform_data,
                        *c,
                        false};
                    DecodeCmd(shader->captured(), visitor);
                    DecodeCmd(c->arguments(), visitor);
                    sizes.second = uniform_data->size() - sizes.first;
                    dispatch_offsets->emplace_back(sizes);
                } break;
                case Command::Tag::ETextureUploadCommand: {
                    auto c = static_cast<TextureUploadCommand const *>(cmd);
                    resource_barrier->record(
                        TexView{
                            reinterpret_cast<Texture const *>(c->handle()),
                            c->level()},
                        ResourceBarrier::Usage::CopyDest);
                } break;
                case Command::Tag::ETextureDownloadCommand: {
                    auto c = static_cast<TextureDownloadCommand const *>(cmd);
                    resource_barrier->record(
                        TexView{
                            reinterpret_cast<Texture const *>(c->handle()),
                            c->level()},
                        ResourceBarrier::Usage::CopySource);
                } break;
                case Command::Tag::ETextureCopyCommand: {
                    auto c = static_cast<TextureCopyCommand const *>(cmd);
                    resource_barrier->record(
                        TexView(
                            reinterpret_cast<Texture const *>(c->src_handle()),
                            c->src_level()),
                        ResourceBarrier::Usage::CopySource);
                    resource_barrier->record(
                        TexView(
                            reinterpret_cast<Texture const *>(c->dst_handle()),
                            c->dst_level()),
                        ResourceBarrier::Usage::CopyDest);
                } break;
                case Command::Tag::ETextureToBufferCopyCommand: {
                    auto c = static_cast<TextureToBufferCopyCommand const *>(cmd);
                    resource_barrier->record(
                        TexView{
                            reinterpret_cast<Texture const *>(c->texture()),
                            c->level()},
                        ResourceBarrier::Usage::CopyDest);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->buffer()),
                            c->buffer_offset(),
                            pixel_storage_size(c->storage(), c->size())},
                        ResourceBarrier::Usage::CopySource);
                } break;
                case Command::Tag::EAccelBuildCommand: {
                    auto c = static_cast<AccelBuildCommand const *>(cmd);
                    reinterpret_cast<Tlas *>(c->handle())->pre_build(*this, c->instance_count(), *write_desc_sets, *bindless_cache, c->modifications(), c->request());
                } break;
                case Command::Tag::EMeshBuildCommand: {
                    auto c = static_cast<MeshBuildCommand const *>(cmd);
                    reinterpret_cast<Blas *>(c->handle())->pre_build(*this, c);
                } break;
                case Command::Tag::ECurveBuildCommand: {
                } break;
                case Command::Tag::EProceduralPrimitiveBuildCommand: {
                } break;
                case Command::Tag::EBindlessArrayUpdateCommand: {
                    auto c = static_cast<BindlessArrayUpdateCommand const *>(cmd);
                    reinterpret_cast<BindlessArray *>(c->handle())->pre_update(resource_barrier);

                } break;
                default: break;
            }
        }
        auto arg_buffer = _state->upload_alloc.allocate(uniform_data->size(), 16);
        static_cast<UploadBuffer const *>(arg_buffer.buffer)->copy_from(uniform_data->data(), arg_buffer.offset, uniform_data->size());
        resource_barrier->update_states(_cmdbuffer);
        auto offset_ptr = dispatch_offsets->data();
        // Execute
        for (auto i = lst; i != nullptr; i = i->p_next) {
            auto cmd = i->cmd;
            switch (cmd->tag()) {
                case Command::Tag::EBufferUploadCommand: {
                    auto c = static_cast<BufferUploadCommand const *>(cmd);
                    auto chunk = _state->upload_alloc.allocate(c->size(), 16);
                    static_cast<UploadBuffer const *>(chunk.buffer)->copy_from(c->data(), chunk.offset, c->size());
                    VkBufferCopy2 buffer_copy{
                        VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                        nullptr,
                        chunk.offset,
                        c->offset(),
                        c->size()};
                    VkCopyBufferInfo2 copy_info2{
                        VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                        nullptr,
                        chunk.buffer->vk_buffer(),
                        reinterpret_cast<DefaultBuffer const *>(c->handle())->vk_buffer(),
                        1,
                        &buffer_copy};
                    vkCmdCopyBuffer2(
                        _cmdbuffer,
                        &copy_info2);
                } break;
                case Command::Tag::EBufferDownloadCommand: {
                    auto c = static_cast<BufferDownloadCommand const *>(cmd);
                    auto chunk = _state->readback_alloc.allocate(c->size(), 16);
                    _state->_callbacks.emplace_back([chunk, data = c->data(), size = c->size()]() {
                        static_cast<ReadbackBuffer const *>(chunk.buffer)->copy_to(data, chunk.offset, size);
                    });
                    VkBufferCopy2 buffer_copy{
                        VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                        nullptr,
                        c->offset(),
                        chunk.offset,
                        c->size()};
                    VkCopyBufferInfo2 copy_info2{
                        VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                        nullptr,
                        reinterpret_cast<DefaultBuffer const *>(c->handle())->vk_buffer(),
                        chunk.buffer->vk_buffer(),
                        1,
                        &buffer_copy};
                    vkCmdCopyBuffer2(
                        _cmdbuffer,
                        &copy_info2);
                } break;
                case Command::Tag::EBufferCopyCommand: {
                    auto c = static_cast<BufferCopyCommand const *>(cmd);
                    VkBufferCopy2 buffer_copy{
                        VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                        nullptr,
                        c->src_offset(),
                        c->dst_offset(),
                        c->size()};
                    VkCopyBufferInfo2 copy_info2{
                        VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                        nullptr,
                        reinterpret_cast<DefaultBuffer const *>(c->src_handle())->vk_buffer(),
                        reinterpret_cast<DefaultBuffer const *>(c->dst_handle())->vk_buffer(),
                        1,
                        &buffer_copy};
                    vkCmdCopyBuffer2(
                        _cmdbuffer,
                        &copy_info2);
                } break;
                case Command::Tag::EBufferToTextureCopyCommand: {
                    auto c = static_cast<BufferToTextureCopyCommand const *>(cmd);
                    auto tex = reinterpret_cast<Texture const *>(c->texture());
                    int3 tex_offset = make_int3(c->texture_offset());
                    auto size = c->size();
                    VkBufferImageCopy2 region{
                        VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                        nullptr,
                        c->buffer_offset(),
                        0,
                        0,
                        VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, c->level(), 0, 1},
                        VkOffset3D{tex_offset.x, tex_offset.y, tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};

                    VkCopyBufferToImageInfo2 copy_info{
                        VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
                        nullptr,
                        reinterpret_cast<DefaultBuffer const *>(c->buffer())->vk_buffer(),
                        tex->vk_image(),
                        resource_barrier->get_layout(tex, c->level()),
                        1,
                        &region};
                    vkCmdCopyBufferToImage2(_cmdbuffer, &copy_info);
                } break;
                case Command::Tag::EShaderDispatchCommand: {
                    auto c = static_cast<ShaderDispatchCommand const *>(cmd);
                    auto shader = reinterpret_cast<ComputeShader *>(c->handle());
                    uint desc_index = 0;

                    BindPropVisitor visitor;
                    visitor.cmdbuffer = this;
                    VkDescriptorSetAllocateInfo alloc_info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                        .descriptorPool = _state->_desc_pool,
                        .descriptorSetCount = 1,
                        .pSetLayouts = shader->desc_set_layout().data()};
                    VK_CHECK_RESULT(
                        vkAllocateDescriptorSets(
                            device()->logic_device(),
                            &alloc_info,
                            &visitor.desc_set));
                    if (offset_ptr->second > 0) {
                        auto arg_buffer_info = temp_desc->allocate_memory<VkDescriptorBufferInfo>();
                        *arg_buffer_info = VkDescriptorBufferInfo{
                            arg_buffer.buffer->vk_buffer(),
                            arg_buffer.offset + offset_ptr->first,
                            offset_ptr->second};
                        write_desc_sets->emplace_back(VkWriteDescriptorSet{
                            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            nullptr,
                            visitor.desc_set,
                            desc_index++,
                            0,
                            1,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            nullptr,
                            arg_buffer_info,
                            nullptr});
                    }
                    visitor.desc_index = desc_index;
                    visitor.img_views = &_state->img_views;
                    auto size = shader->saved_arguments().size();

                    visitor.arg = shader->saved_arguments().data();
                    DecodeCmd(shader->captured(), visitor);
                    DecodeCmd(c->arguments(), visitor);
                    if (!write_desc_sets->empty()) {
                        vkUpdateDescriptorSets(
                            device()->logic_device(),
                            write_desc_sets->size(),
                            write_desc_sets->data(), 0,
                            nullptr);
                        write_desc_sets->clear();
                    }
                    desc_sets->clear();
                    desc_sets->push_back(visitor.desc_set);
                    desc_sets->push_back(device()->sampler_set());
                    if (shader->use_buffer_bindless()) {
                        desc_sets->push_back(device()->bdls_buffer_set());
                    }
                    if (shader->use_tex2d_bindless()) {
                        desc_sets->push_back(device()->bdls_tex2d_set());
                    }
                    if (shader->use_tex3d_bindless()) {
                        desc_sets->push_back(device()->bdls_tex3d_set());
                    }
                    vkCmdBindDescriptorSets(
                        _cmdbuffer,
                        VK_PIPELINE_BIND_POINT_COMPUTE,
                        shader->pipeline_layout(),
                        0,
                        desc_sets->size(),
                        desc_sets->data(),
                        0,
                        nullptr);
                    vkCmdBindPipeline(_cmdbuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shader->pipeline());
                    auto calc = [](uint disp, uint thd) {
                        return (disp + thd - 1u) / thd;
                    };
                    auto blk = shader->block_size();
                    if (c->is_multiple_dispatch()) {
                        LUISA_ASSERT(false, "Dispatch count not implemented.");
                        uint idx = 0;
                        for (auto &disp_size : c->dispatch_sizes()) {
                            uint4 value{make_uint4(disp_size, idx)};
                            ++idx;
                            vkCmdPushConstants(
                                _cmdbuffer,
                                shader->pipeline_layout(),
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0,
                                16,
                                &value);
                            vkCmdDispatch(_cmdbuffer, calc(disp_size.x, blk.x), calc(disp_size.y, blk.y), calc(disp_size.z, blk.z));
                        }
                    } else {
                        auto disp_size = c->dispatch_size();
                        uint4 value{make_uint4(disp_size, 0)};
                        vkCmdPushConstants(
                            _cmdbuffer,
                            shader->pipeline_layout(),
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            16,
                            &value);
                        vkCmdDispatch(_cmdbuffer, calc(disp_size.x, blk.x), calc(disp_size.y, blk.y), calc(disp_size.z, blk.z));
                    }
                    offset_ptr++;
                } break;
                case Command::Tag::ETextureUploadCommand: {
                    auto c = static_cast<TextureUploadCommand const *>(cmd);
                    auto pixel_size = pixel_storage_size(c->storage(), c->size());
                    auto buffer = _state->upload_alloc.allocate(pixel_size, 16);
                    static_cast<UploadBuffer const *>(buffer.buffer)->copy_from(c->data(), buffer.offset, pixel_size);
                    auto tex = reinterpret_cast<Texture const *>(c->handle());
                    int3 tex_offset = make_int3(c->offset());
                    auto size = c->size();
                    VkBufferImageCopy2 region{
                        VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                        nullptr,
                        buffer.offset,
                        0,
                        0,
                        VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, c->level(), 0, 1},
                        VkOffset3D{tex_offset.x, tex_offset.y, tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};

                    VkCopyBufferToImageInfo2 copy_info{
                        VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
                        nullptr,
                        buffer.buffer->vk_buffer(),
                        tex->vk_image(),
                        resource_barrier->get_layout(tex, c->level()),
                        1,
                        &region};
                    vkCmdCopyBufferToImage2(_cmdbuffer, &copy_info);
                } break;
                case Command::Tag::ETextureDownloadCommand: {
                    auto c = static_cast<TextureDownloadCommand const *>(cmd);
                    auto pixel_size = pixel_storage_size(c->storage(), c->size());
                    auto buffer = _state->readback_alloc.allocate(pixel_size, 16);
                    _state->_callbacks.emplace_back([buffer = buffer.buffer,
                                                     offset = buffer.offset,
                                                     pixel_size,
                                                     data = c->data()]() {
                        static_cast<ReadbackBuffer const *>(buffer)->copy_to(data, offset, pixel_size);
                    });
                    auto tex = reinterpret_cast<Texture const *>(c->handle());
                    int3 tex_offset = make_int3(c->offset());
                    auto size = c->size();
                    VkBufferImageCopy2 region{
                        VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                        nullptr,
                        buffer.offset,
                        0,
                        0,
                        VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, c->level(), 0, 1},
                        VkOffset3D{tex_offset.x, tex_offset.y, tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};
                    VkCopyImageToBufferInfo2 info{
                        VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
                        nullptr,
                        tex->vk_image(),
                        resource_barrier->get_layout(tex, c->level()),
                        buffer.buffer->vk_buffer(),
                        1,
                        &region};
                    vkCmdCopyImageToBuffer2(
                        _cmdbuffer,
                        &info);
                } break;
                case Command::Tag::ETextureCopyCommand: {
                    auto c = static_cast<TextureCopyCommand const *>(cmd);
                    auto src_tex = reinterpret_cast<Texture const *>(c->src_handle());
                    int3 src_tex_offset = make_int3(c->src_offset());
                    int3 dst_tex_offset = make_int3(c->dst_offset());
                    auto dst_tex = reinterpret_cast<Texture const *>(c->dst_handle());
                    auto size = c->size();
                    VkImageCopy2 copy{
                        VK_STRUCTURE_TYPE_IMAGE_COPY_2,
                        nullptr,
                        VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, c->src_level(), 0, 1},
                        VkOffset3D{src_tex_offset.x, src_tex_offset.y, src_tex_offset.z},
                        VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, c->dst_level(), 0, 1},
                        VkOffset3D{dst_tex_offset.x, dst_tex_offset.y, dst_tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};
                    VkCopyImageInfo2 info{
                        VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
                        nullptr,
                        src_tex->vk_image(),
                        resource_barrier->get_layout(src_tex, c->src_level()),
                        dst_tex->vk_image(),
                        resource_barrier->get_layout(dst_tex, c->src_level()),
                        1,
                        &copy};
                    vkCmdCopyImage2(
                        _cmdbuffer,
                        &info);
                } break;
                case Command::Tag::ETextureToBufferCopyCommand: {
                    auto c = static_cast<TextureToBufferCopyCommand const *>(cmd);
                    auto tex = reinterpret_cast<Texture const *>(c->texture());
                    int3 tex_offset = make_int3(c->texture_offset());
                    auto size = c->size();
                    VkBufferImageCopy2 region{
                        VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                        nullptr,
                        c->buffer_offset(),
                        0,
                        0,
                        VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, c->level(), 0, 1},
                        VkOffset3D{tex_offset.x, tex_offset.y, tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};
                    VkCopyImageToBufferInfo2 info{
                        VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
                        nullptr,
                        tex->vk_image(),
                        resource_barrier->get_layout(tex, c->level()),
                        reinterpret_cast<DefaultBuffer const *>(c->buffer())->vk_buffer(),
                        1,
                        &region};
                    vkCmdCopyImageToBuffer2(
                        _cmdbuffer,
                        &info);
                } break;
                case Command::Tag::EAccelBuildCommand: {
                    auto c = static_cast<AccelBuildCommand const *>(cmd);
                    reinterpret_cast<Tlas *>(c->handle())->build(*this, c->instance_count());
                    auto &bf = *reinterpret_cast<Tlas *>(c->handle())->instance_buffer();
                    // resource_barrier->record(
                    //     BufferView{&bf},
                    //     ResourceBarrier::Usage::CopySource);
                    // resource_barrier->update_states(_cmdbuffer);
                    // luisa::vector<VkAccelerationStructureInstanceKHR> vec(bf.byte_size() / sizeof(VkAccelerationStructureInstanceKHR));
                    // auto chunk = _state->readback_alloc.allocate(vec.size_bytes(), 16);

                    // VkBufferCopy2 buffer_copy{
                    //     VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                    //     nullptr,
                    //     0,
                    //     chunk.offset,
                    //     vec.size_bytes()};
                    // int x = 0;
                    // VkCopyBufferInfo2 copy_info2{
                    //     VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                    //     nullptr,
                    //     bf.vk_buffer(),
                    //     chunk.buffer->vk_buffer(),
                    //     1,
                    //     &buffer_copy};
                    // vkCmdCopyBuffer2(
                    //     _cmdbuffer,
                    //     &copy_info2);
                    // _state->_callbacks.emplace_back([chunk, vec = std::move(vec)]() mutable {
                    //     static_cast<ReadbackBuffer const *>(chunk.buffer)->copy_to(vec.data(), chunk.offset, vec.size_bytes());
                    //     for (auto &i : vec) {
                    //         LUISA_INFO(
                    //             "Matrix: {}\n{}\n{}\ninstanceCustomIndex{}\nmask{}\ninstanceShaderBindingTableRecordOffset{}\nflags{}\naccelerationStructureReference{}",
                    //             (float4&)i.transform.matrix[0],
                    //             (float4&)i.transform.matrix[1],
                    //             (float4&)i.transform.matrix[2],
                    //             (uint)i.instanceCustomIndex,
                    //             (uint)i.mask,
                    //             (uint)i.instanceShaderBindingTableRecordOffset,
                    //             (uint)i.flags,
                    //             i.accelerationStructureReference
                    //         );
                    //     }
                    // });
                } break;
                case Command::Tag::EMeshBuildCommand: {
                    auto c = static_cast<MeshBuildCommand const *>(cmd);
                    reinterpret_cast<Blas *>(c->handle())->build(*this, c);
                } break;
                case Command::Tag::ECurveBuildCommand: {
                } break;
                case Command::Tag::EProceduralPrimitiveBuildCommand: {
                } break;
                case Command::Tag::EBindlessArrayUpdateCommand: {
                    auto c = static_cast<BindlessArrayUpdateCommand const *>(cmd);
                    reinterpret_cast<BindlessArray *>(c->handle())->update(this, *write_desc_sets, *bindless_cache, c->modifications());
                    // LOG bindless indices

                    // auto &bf = reinterpret_cast<BindlessArray *>(c->handle())->indices_buffer();
                    // resource_barrier->record(
                    //     BufferView{&bf},
                    //     ResourceBarrier::Usage::CopySource);
                    // resource_barrier->update_states(_cmdbuffer);

                    // luisa::vector<std::array<uint, 3>> vec(16);
                    // auto chunk = _state->readback_alloc.allocate(vec.size(), 16);

                    // VkBufferCopy2 buffer_copy{
                    //     VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                    //     nullptr,
                    //     0,
                    //     chunk.offset,
                    //     vec.size_bytes()};
                    // int x = 0;
                    // VkCopyBufferInfo2 copy_info2{
                    //     VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                    //     nullptr,
                    //     bf.vk_buffer(),
                    //     chunk.buffer->vk_buffer(),
                    //     1,
                    //     &buffer_copy};
                    // vkCmdCopyBuffer2(
                    //     _cmdbuffer,
                    //     &copy_info2);
                    // _state->_callbacks.emplace_back([chunk, vec = std::move(vec)]() mutable {
                    //     static_cast<ReadbackBuffer const *>(chunk.buffer)->copy_to(vec.data(), chunk.offset, vec.size_bytes());
                    //     for (auto &i : vec) {
                    //         LUISA_INFO(uint3(i[0], i[1], i[2]));
                    //     }
                    // });
                } break;
                default: break;
            }
        }
    }
    resource_barrier->restore_states(_cmdbuffer);

    temp_desc->clear();
}

vstd::span<VkDescriptorSet> Shader::allocate_desc_set(VkDescriptorPool pool, vstd::vector<VkDescriptorSet> &descs) const {
    VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = static_cast<uint>(_desc_set_layout.size()),
        .pSetLayouts = _desc_set_layout.data()};
    auto last_size = descs.size();
    descs.push_back_uninitialized(_desc_set_layout.size());
    VK_CHECK_RESULT(
        vkAllocateDescriptorSets(
            device()->logic_device(),
            &alloc_info,
            descs.data() + last_size));
    return vstd::span<VkDescriptorSet>{descs.data() + last_size, _desc_set_layout.size()};
}
void Shader::update_desc_set(
    VkDescriptorSet set,
    vstd::vector<VkWriteDescriptorSet> &write_buffer,
    vstd::vector<VkImageView> &img_view_buffer,
    vstd::span<vstd::variant<BufferView, TexView>> texs) {
    write_buffer.clear();
    write_buffer.reserve(texs.size());
    uint arg_idx = 0;
    VkDescriptorBufferInfo buffer_info;
    VkDescriptorImageInfo image_info;
    auto make_desc = [&]<typename T>(T const &t) {
        auto &v = write_buffer.emplace_back();
        v.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        v.dstSet = set;
        v.dstBinding = arg_idx;
        v.dstArrayElement = 1;
        v.descriptorCount = 1;
        auto &&b = _binds[arg_idx];

        switch (b.type) {
            case lc::hlsl::ShaderVariableType::ConstantBuffer:
                v.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case lc::hlsl::ShaderVariableType::SRVTextureHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                break;
            case lc::hlsl::ShaderVariableType::UAVTextureHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                break;
            case lc::hlsl::ShaderVariableType::SRVBufferHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case lc::hlsl::ShaderVariableType::UAVBufferHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case lc::hlsl::ShaderVariableType::CBVBufferHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case lc::hlsl::ShaderVariableType::SamplerHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                break;
            case lc::hlsl::ShaderVariableType::StructuredBuffer:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case lc::hlsl::ShaderVariableType::RWStructuredBuffer:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case lc::hlsl::ShaderVariableType::ConstantValue:
                v.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
        }
        if constexpr (std::is_same_v<T, Argument::Buffer>) {
            buffer_info.buffer = reinterpret_cast<Buffer *>(t.handle)->vk_buffer();
            buffer_info.offset = t.offset;
            buffer_info.range = t.size;
            v.pBufferInfo = &buffer_info;
        }
        if constexpr (std::is_same_v<T, Argument::Texture>) {
            image_info.sampler = nullptr;
            auto &img_view = img_view_buffer.emplace_back();
            auto tex = reinterpret_cast<Texture *>(t.handle);
            VkImageViewCreateInfo img_view_create_info = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .flags = 0,
                .image = tex->vk_image(),
                .viewType = [&]() {
                    switch (tex->dimension()) {
                        case 1:
                            return VK_IMAGE_VIEW_TYPE_1D;
                        case 2:
                            return VK_IMAGE_VIEW_TYPE_2D;
                        case 3:
                            return VK_IMAGE_VIEW_TYPE_3D;
                    }
                }(),
                .format = Texture::to_vk_format(tex->format()),
                .subresourceRange = VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = t.level, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
            VK_CHECK_RESULT(vkCreateImageView(device()->logic_device(), &img_view_create_info, Device::alloc_callbacks(), &img_view));
            image_info.imageView = img_view;
            image_info.imageLayout = tex->layout(t.level);
            v.pImageInfo = &image_info;
        }
        arg_idx++;
    };
    for (auto i : vstd::range(texs.size())) {

        // v.descriptorType = view.index() ==
    }
}
}// namespace lc::vk
