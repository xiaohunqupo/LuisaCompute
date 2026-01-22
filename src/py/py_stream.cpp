#include "py_stream.h"
#include <luisa/runtime/command_list.h>
namespace luisa::compute {

PyStream::PyStream(Device &device, bool support_window) noexcept
    : _data{luisa::make_shared<Data>(device, support_window)} {}

PyStream::Data::Data(Device &device, bool support_window) noexcept
    : stream(device.create_stream(support_window ? StreamTag::GRAPHICS : StreamTag::COMPUTE)) {
}

PyStream::~PyStream() noexcept {
    if (!_data) return;
    execute();
    _data->stream.synchronize();
}

void PyStream::add(Command *cmd) noexcept {
    _data->buffer << luisa::unique_ptr<Command>(cmd);
}

void PyStream::add(luisa::unique_ptr<Command> &&cmd) noexcept {
    _data->buffer << std::move(cmd);
}
void PyStream::Data::sync() noexcept {
    execute();
    stream << synchronize();
    uploadDisposer.clear();
}

void PyStream::Data::execute() noexcept {
    if (!buffer.commands().empty()) {
        for (auto &i : before_commit_tasks) {
            i.second(i.first, stream);
        }
        before_commit_tasks.clear();
    }
    if (!buffer.empty()) {
        stream << buffer.commit();
    }
    uploadDisposer.clear();
}

void PyStream::execute() noexcept {
    if (!delegates.empty()) {
        _data->buffer.add_callback([delegates = std::move(delegates)] {
            for (auto &&i : delegates) { i(); }
        });
    }
    _data->execute();
}

void PyStream::sync() noexcept {
    execute();
    _data->stream.synchronize();
}

PyStream::PyStream(PyStream &&s) noexcept
    : _data(std::move(s._data)) {}

}// namespace luisa::compute
