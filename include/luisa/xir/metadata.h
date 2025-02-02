#pragma once

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/xir/ilist.h>

namespace luisa::compute::xir {

enum struct DerivedMetadataTag {
    NAME,
    LOCATION,
    COMMENT,
};

class Metadata : public IntrusiveForwardNode<Metadata> {
public:
    using Super::Super;
    [[nodiscard]] virtual DerivedMetadataTag derived_metadata_tag() const noexcept = 0;
    [[nodiscard]] virtual Metadata *clone() const noexcept = 0;
    LUISA_XIR_DEFINED_ISA_METHOD(Metadata, metadata)
};

template<typename Derived, DerivedMetadataTag tag, typename Base = Metadata>
class LC_XIR_API DerivedMetadata : public Base {
public:
    using derived_metadata_type = Derived;
    using Base::Base;
    [[nodiscard]] static constexpr auto
    static_derived_metadata_tag() noexcept {
        return tag;
    }
    [[nodiscard]] DerivedMetadataTag
    derived_metadata_tag() const noexcept final {
        return static_derived_metadata_tag();
    }
};

using MetadataList = IntrusiveForwardList<Metadata>;

class LC_XIR_API MetadataMixin {

private:
    MetadataList _metadata_list{};

protected:
    MetadataMixin() noexcept = default;
    ~MetadataMixin() noexcept = default;

public:
    [[nodiscard]] auto &metadata_list() noexcept { return _metadata_list; }
    [[nodiscard]] auto &metadata_list() const noexcept { return _metadata_list; }

    [[nodiscard]] Metadata *find_metadata(DerivedMetadataTag tag) noexcept;
    [[nodiscard]] const Metadata *find_metadata(DerivedMetadataTag tag) const noexcept;
    [[nodiscard]] Metadata *create_metadata(DerivedMetadataTag tag) noexcept;
    [[nodiscard]] Metadata *find_or_create_metadata(DerivedMetadataTag tag) noexcept;

    template<typename T>
    [[nodiscard]] auto find_metadata() noexcept {
        return static_cast<T *>(find_metadata(T::static_derived_metadata_tag()));
    }
    template<typename T>
    [[nodiscard]] auto find_metadata() const noexcept {
        return static_cast<const T *>(find_metadata(T::static_derived_metadata_tag()));
    }
    template<typename T>
    [[nodiscard]] auto create_metadata() noexcept {
        return static_cast<T *>(create_metadata(T::static_derived_metadata_tag()));
    }
    template<typename T>
    [[nodiscard]] auto find_or_create_metadata() noexcept {
        return static_cast<T *>(find_or_create_metadata(T::static_derived_metadata_tag()));
    }

    void set_name(std::string_view name) noexcept;
    void set_location(const std::filesystem::path &file, int line = -1) noexcept;
    void add_comment(std::string_view comment) noexcept;

    [[nodiscard]] luisa::optional<luisa::string_view> name() const noexcept;
};

}// namespace luisa::compute::xir
