#include <fstream>

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/event.h>
#include <luisa/backends/ext/dstorage_ext.hpp>
#include "../../reference_image.h"
#include <luisa/core/clock.h>

#include <filesystem>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    auto opts = luisa::test::ImageTestOptions::parse(argc, argv);
    auto device = context.create_device(argv[1]);
    auto dstorage_ext = device.extension<DStorageExt>();

    auto dstorage_stream = dstorage_ext->create_stream();
    auto dstorage_file = dstorage_ext->open_file("test_dstorage_texture_compressed.gdeflate");
    auto image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(4096));
    dstorage_stream << dstorage_file.copy_to(image, DStorageCompression::GDeflate) << synchronize();

    luisa::vector<uint8_t> pixels(image.view().size_bytes());
    auto compute_stream = device.create_stream();
    compute_stream << image.copy_to(pixels.data()) << synchronize();

    stbi_write_png("test_dstorage_decompression.png", 4096, 4096, 4, pixels.data(), 0);
    auto ref_dir = luisa::test::find_reference_dir(std::filesystem::path{argv[0]}.parent_path());
    auto result = luisa::test::save_and_compare(
        pixels.data(), 4096, 4096, 4,
        "test_dstorage_decompression", opts.output_dir, ref_dir, opts.update_reference);
    LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
    if (!result.passed) {
        LUISA_ERROR("Reference comparison failed for test_dstorage_decompression: {}", result.message);
        if (opts.offline) { return 1; }
        return 1;
    }
    return 0;
}
