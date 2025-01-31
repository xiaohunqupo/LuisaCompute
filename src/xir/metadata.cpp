#include <luisa/core/logging.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/xir/metadata/location.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/metadata.h>

namespace luisa::compute::xir {

Metadata *MetadataMixin::find_metadata(DerivedMetadataTag tag) noexcept {
    for (auto &m : _metadata_list) {
        if (m.derived_metadata_tag() == tag) { return &m; }
    }
    return nullptr;
}

const Metadata *MetadataMixin::find_metadata(DerivedMetadataTag tag) const noexcept {
    return const_cast<MetadataMixin *>(this)->find_metadata(tag);
}

Metadata *MetadataMixin::create_metadata(DerivedMetadataTag tag) noexcept {
    auto pool = Pool::current();
    switch (tag) {
#define LUISA_XIR_MAKE_METADATA_CREATE_CASE(type)   \
    case type##MD::static_derived_metadata_tag(): { \
        auto m = pool->create<type##MD>();          \
        m->add_to_list(_metadata_list);             \
        return m;                                   \
    }
        LUISA_XIR_MAKE_METADATA_CREATE_CASE(Name)
        LUISA_XIR_MAKE_METADATA_CREATE_CASE(Location)
        LUISA_XIR_MAKE_METADATA_CREATE_CASE(Comment)
#undef LUISA_XIR_MAKE_METADATA_CREATE_CASE
    }
    LUISA_ERROR_WITH_LOCATION("Unknown derived metadata tag 0x{:x}.",
                              static_cast<uint32_t>(tag));
}

Metadata *MetadataMixin::find_or_create_metadata(DerivedMetadataTag tag) noexcept {
    if (auto m = find_metadata(tag); m != nullptr) { return m; }
    return create_metadata(tag);
}

void MetadataMixin::set_name(std::string_view name) noexcept {
    auto m = find_or_create_metadata<NameMD>();
    m->set_name(name);
}

std::optional<luisa::string> MetadataMixin::name() const noexcept {
    if (auto m = find_metadata<NameMD>(); m != nullptr) { return m->name(); }
    return std::nullopt;
}

void MetadataMixin::set_location(const std::filesystem::path &file, int line) noexcept {
    auto m = find_or_create_metadata<LocationMD>();
    m->set_location(file, line);
}

void MetadataMixin::add_comment(std::string_view comment) noexcept {
    auto m = create_metadata<CommentMD>();
    m->set_comment(comment);
}

}// namespace luisa::compute::xir
