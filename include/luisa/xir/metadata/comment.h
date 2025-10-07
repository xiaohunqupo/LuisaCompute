#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/xir/metadata.h>

namespace luisa::compute::xir {

class LUISA_XIR_API CommentMD final : public DerivedMetadata<CommentMD, DerivedMetadataTag::COMMENT> {

private:
    luisa::string _comment;

public:
    explicit CommentMD(luisa::string comment = {}) noexcept;
    [[nodiscard]] auto &comment() noexcept { return _comment; }
    [[nodiscard]] auto &comment() const noexcept { return _comment; }
    void set_comment(luisa::string_view comment) noexcept;
    [[nodiscard]] ManagedPtr<Metadata> clone() const noexcept override;
};

}// namespace luisa::compute::xir
