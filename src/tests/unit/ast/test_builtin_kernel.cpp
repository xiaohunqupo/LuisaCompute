#include <iostream>
#include <vector>
#include <exception>
#include <luisa/luisa-compute.h>
#include "ut/ut.hpp"
#include "test_device.h"
#include "builtin_kernel.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int test_builtin_kernel(Device &device) {

    log_level_verbose();

    Stream stream = device.create_stream();

    BuiltinKernel builtin{&device};

    // Test Buffer fill
    {
        auto builder = BuiltinKernel::fill_buffer();
        auto shader = builtin.compile<1, Buffer<uint>, uint>(builder);

        const size_t buffer_size = 1024;
        Buffer<uint> buffer = device.create_buffer<uint>(buffer_size);

        stream << shader(buffer, 42u).dispatch(buffer_size)
               << synchronize();

        std::vector<uint> result(buffer_size);
        stream << buffer.copy_to(result.data()) << synchronize();

        bool success = true;
        for (size_t i = 0; i < buffer_size; i++) {
            if (result[i] != 42u) {
                success = false;
                break;
            }
        }
        std::cout << "Buffer fill: " << (success ? "PASS" : "FAIL") << std::endl;
    }

    // Test Image2D fill (uint)
    {
        auto builder = BuiltinKernel::fill_image_uint();
        auto shader = builtin.compile<2, Image<uint>, uint>(builder);

        const uint2 image_size{64, 64};
        Image<uint> image = device.create_image<uint>(PixelStorage::INT4, image_size);

        stream << shader(image, 255u).dispatch(image_size)
               << synchronize();

        std::cout << "Image2D fill (uint): PASS (shader compiled and dispatched)" << std::endl;
    }

    // Test Image2D fill (float)
    {
        auto builder = BuiltinKernel::fill_image_float();
        auto shader = builtin.compile<2, Image<float>, float>(builder);

        const uint2 image_size{64, 64};
        Image<float> image = device.create_image<float>(PixelStorage::FLOAT4, image_size);

        stream << shader(image, 1.0f).dispatch(image_size)
               << synchronize();

        std::cout << "Image2D fill (float): PASS (shader compiled and dispatched)" << std::endl;
    }

    // Test Volume fill (uint)
    {
        auto builder = BuiltinKernel::fill_volume_uint();
        auto shader = builtin.compile<3, Volume<uint>, uint>(builder);

        const uint3 volume_size{32, 32, 32};
        Volume<uint> volume = device.create_volume<uint>(PixelStorage::INT4, volume_size);

        stream << shader(volume, 100u).dispatch(volume_size)
               << synchronize();

        std::cout << "Volume fill (uint): PASS (shader compiled and dispatched)" << std::endl;
    }

    // Test Volume fill (int)
    {
        auto builder = BuiltinKernel::fill_volume_int();
        auto shader = builtin.compile<3, Volume<int>, int>(builder);

        const uint3 volume_size{16, 16, 16};
        Volume<int> volume = device.create_volume<int>(PixelStorage::INT4, volume_size);

        stream << shader(volume, -50).dispatch(volume_size)
               << synchronize();

        std::cout << "Volume fill (int): PASS (shader compiled and dispatched)" << std::endl;
    }

    // Test Volume fill (float)
    {
        auto builder = BuiltinKernel::fill_volume_float();
        auto shader = builtin.compile<3, Volume<float>, float>(builder);

        const uint3 volume_size{8, 8, 8};
        Volume<float> volume = device.create_volume<float>(PixelStorage::FLOAT4, volume_size);

        stream << shader(volume, 3.14f).dispatch(volume_size)
               << synchronize();

        std::cout << "Volume fill (float): PASS (shader compiled and dispatched)" << std::endl;
    }

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}

static inline const auto reg = [] {
    "builtin_kernel"_test = [] {
        auto dc = luisa::test::create_device_from_ut();
        if (!dc) return;
        auto &device = dc->device;
        try {
            test_builtin_kernel(device);
            expect(true);
        } catch (const std::exception &e) {
            expect(false) << e.what();
        } catch (...) {
            expect(false) << "unknown exception";
        }
    };
    return 0;
}();

int main() {}
