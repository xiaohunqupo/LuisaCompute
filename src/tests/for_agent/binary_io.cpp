// Test for binary_io.h
// Tests BinaryBlob class for memory management, data access, and move semantics.

#include <cstring>

#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../common/doctest.h"

#include <luisa/core/binary_io.h>
#include <luisa/core/logging.h>

using namespace luisa;

// Custom test allocator tracking
namespace {
    int g_allocate_count = 0;
    int g_dispose_count = 0;
}

TEST_SUITE("BinaryBlob") {

    TEST_CASE("BinaryBlob default construction") {
        BinaryBlob blob;
        CHECK(blob.data() == nullptr);
        CHECK(blob.size() == 0);
        CHECK(blob.empty() == true);
    }

    TEST_CASE("BinaryBlob construction with data") {
        g_allocate_count = 0;
        g_dispose_count = 0;
        
        size_t size = 1024;
        auto *ptr = static_cast<std::byte *>(::operator new(size));
        g_allocate_count++;
        
        {
            BinaryBlob blob{
                ptr,
                size,
                [](void *p) { 
                    ::operator delete(p); 
                    g_dispose_count++;
                }};
            
            CHECK(blob.data() == ptr);
            CHECK(blob.size() == size);
            CHECK(blob.empty() == false);
        }
        
        // Destructor should have called disposer
        CHECK(g_dispose_count == 1);
    }

    TEST_CASE("BinaryBlob move construction") {
        size_t size = 256;
        auto *ptr = static_cast<std::byte *>(::operator new(size));
        
        BinaryBlob blob1{
            ptr,
            size,
            [](void *p) { ::operator delete(p); }};
        
        BinaryBlob blob2{std::move(blob1)};
        
        // blob1 should be empty after move
        CHECK(blob1.data() == nullptr);
        CHECK(blob1.size() == 0);
        CHECK(blob1.empty() == true);
        
        // blob2 should have the data
        CHECK(blob2.data() == ptr);
        CHECK(blob2.size() == size);
        CHECK(blob2.empty() == false);
    }

    TEST_CASE("BinaryBlob move assignment") {
        size_t size1 = 128;
        size_t size2 = 256;
        auto *ptr1 = static_cast<std::byte *>(::operator new(size1));
        auto *ptr2 = static_cast<std::byte *>(::operator new(size2));
        int dispose_count = 0;
        
        {
            BinaryBlob blob1{
                ptr1,
                size1,
                [&dispose_count](void *p) { 
                    ::operator delete(p); 
                    dispose_count++;
                }};
            
            BinaryBlob blob2{
                ptr2,
                size2,
                [&dispose_count](void *p) { 
                    ::operator delete(p); 
                    dispose_count++;
                }};
            
            blob1 = std::move(blob2);
            
            // blob1's old data should be disposed
            CHECK(dispose_count == 1);
            
            // blob1 should now have blob2's data
            CHECK(blob1.data() == ptr2);
            CHECK(blob1.size() == size2);
        }
        
        // Both blobs disposed
        CHECK(dispose_count == 2);
    }

    TEST_CASE("BinaryBlob const and non-const data access") {
        size_t size = 64;
        auto *ptr = static_cast<std::byte *>(::operator new(size));
        
        // Non-const blob
        {
            BinaryBlob blob{
                ptr,
                size,
                [](void *p) { ::operator delete(p); }};
            
            std::byte *data = blob.data();
            CHECK(data == ptr);
            
            // Modify through non-const pointer
            data[0] = std::byte{0x42};
            CHECK(static_cast<int>(blob.data()[0]) == 0x42);
        }
        
        // Const blob
        {
            auto *ptr2 = static_cast<std::byte *>(::operator new(size));
            const BinaryBlob blob{
                ptr2,
                size,
                [](void *p) { ::operator delete(p); }};
            
            const std::byte *data = blob.data();
            CHECK(data == ptr2);
        }
    }

    TEST_CASE("BinaryBlob span conversion") {
        size_t size = 32;
        auto data = new std::byte[size];
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<std::byte>(i);
        }
        
        {
            BinaryBlob blob{
                data,
                size,
                [](void *p) { delete[] static_cast<std::byte *>(p); }};
            
            // Non-const span
            luisa::span<std::byte> mutable_span = static_cast<luisa::span<std::byte>>(blob);
            CHECK(mutable_span.size() == size);
            CHECK(mutable_span.data() == data);
            
            // Modify through span
            mutable_span[0] = std::byte{0xFF};
            CHECK(static_cast<int>(blob.data()[0]) == 0xFF);
        }
        
        // Const span test
        auto data2 = new std::byte[size];
        {
            const BinaryBlob blob{
                data2,
                size,
                [](void *p) { delete[] static_cast<std::byte *>(p); }};
            
            luisa::span<const std::byte> const_span = static_cast<luisa::span<const std::byte>>(blob);
            CHECK(const_span.size() == size);
            CHECK(const_span.data() == data2);
        }
    }

    TEST_CASE("BinaryBlob empty check") {
        BinaryBlob empty_blob;
        CHECK(empty_blob.empty() == true);
        
        auto *ptr = static_cast<std::byte *>(::operator new(1));
        BinaryBlob non_empty_blob{
            ptr,
            1,
            [](void *p) { ::operator delete(p); }};
        CHECK(non_empty_blob.empty() == false);
    }

    TEST_CASE("BinaryBlob self-move assignment") {
        size_t size = 64;
        auto *ptr = static_cast<std::byte *>(::operator new(size));
        int dispose_count = 0;
        
        BinaryBlob blob{
            ptr,
            size,
            [&dispose_count](void *p) { 
                ::operator delete(p); 
                dispose_count++;
            }};
        
        // Self-move assignment (should be safe)
        blob = std::move(blob);
        
        // Data should still be valid
        CHECK(blob.data() == ptr);
        CHECK(blob.size() == size);
        CHECK(dispose_count == 0);
    }

    TEST_CASE("BinaryBlob multiple moves") {
        size_t size = 100;
        auto *ptr = static_cast<std::byte *>(::operator new(size));
        int dispose_count = 0;
        
        {
            BinaryBlob blob1{
                ptr,
                size,
                [&dispose_count](void *p) { 
                    ::operator delete(p); 
                    dispose_count++;
                }};
            
            BinaryBlob blob2{std::move(blob1)};
            BinaryBlob blob3{std::move(blob2)};
            BinaryBlob blob4{std::move(blob3)};
            
            CHECK(blob4.data() == ptr);
            CHECK(blob4.size() == size);
            CHECK(blob1.data() == nullptr);
            CHECK(blob2.data() == nullptr);
            CHECK(blob3.data() == nullptr);
        }
        
        CHECK(dispose_count == 1);
    }

    TEST_CASE("BinaryBlob with custom allocator") {
        struct TestAllocator {
            size_t allocated = 0;
            size_t deallocated = 0;
            
            void *allocate(size_t size) {
                allocated += size;
                return std::malloc(size);
            }
            
            void deallocate(void *ptr, size_t size) {
                deallocated += size;
                std::free(ptr);
            }
        };
        
        TestAllocator allocator;
        size_t size = 512;
        {
            auto *ptr = static_cast<std::byte *>(allocator.allocate(size));
            BinaryBlob blob{
                ptr,
                size,
                [&allocator, size](void *p) { allocator.deallocate(p, size); }};
            
            CHECK(allocator.allocated == size);
        }
        
        CHECK(allocator.deallocated == size);
    }
}
