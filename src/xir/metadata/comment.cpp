#include <luisa/xir/metadata/comment.h>

namespace luisa::compute::xir {

CommentMD::CommentMD(luisa::string comment) noexcept
    : _comment{std::move(comment)} {}

void CommentMD::set_comment(luisa::string_view comment) noexcept {
    _comment = comment;
}

ManagedPtr<Metadata> CommentMD::clone() const noexcept {
    return luisa::make_managed<CommentMD>(this->comment());
}

}// namespace luisa::compute::xir
