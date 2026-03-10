// Test for FirstFit memory allocator
// This test covers:
// - Basic construction with size and alignment
// - allocate() - First-fit allocation strategy
// - allocate_best_fit() - Best-fit allocation strategy
// - free() - Deallocation and memory reuse
// - Move semantics (move constructor and move assignment)
// - dump_free_list() - Debugging free list state
// - Fragmentation scenarios
// - Edge cases (exact fit, alignment constraints)

#include <vector>
#include <algorithm>

#include <luisa/core/first_fit.h>
#include <luisa/core/logging.h>

using namespace luisa;

// Test basic construction and simple allocation
void test_basic_construction() {
    LUISA_INFO("Testing basic construction...");

    // Create a FirstFit allocator with 1024 bytes and 8-byte alignment
    FirstFit allocator(1024, 8);

    LUISA_ASSERT(allocator.size() == 1024, "Size mismatch: expected 1024, got {}", allocator.size());
    LUISA_ASSERT(allocator.alignment() == 8, "Alignment mismatch: expected 8, got {}", allocator.alignment());

    LUISA_INFO("Basic construction test passed.");
}

// Test basic allocate and free
void test_basic_allocate_free() {
    LUISA_INFO("Testing basic allocate/free...");

    FirstFit allocator(1024, 8);

    // Allocate some blocks
    auto* node1 = allocator.allocate(100);
    LUISA_ASSERT(node1 != nullptr, "Allocation failed for 100 bytes");
    LUISA_ASSERT(node1->offset() == 0, "First allocation should start at offset 0, got {}", node1->offset());
    LUISA_ASSERT(node1->size() >= 100, "Allocated size should be at least 100, got {}", node1->size());

    auto* node2 = allocator.allocate(200);
    LUISA_ASSERT(node2 != nullptr, "Allocation failed for 200 bytes");
    LUISA_ASSERT(node2->offset() >= node1->offset() + node1->size(), "Second allocation should be after first");

    auto* node3 = allocator.allocate(300);
    LUISA_ASSERT(node3 != nullptr, "Allocation failed for 300 bytes");

    LUISA_INFO("Allocated blocks at offsets: {}, {}, {}", node1->offset(), node2->offset(), node3->offset());

    // Free middle block and reallocate
    size_t old_offset = node2->offset();
    allocator.free(node2);

    // Allocate smaller block - should fit in freed space (first-fit)
    auto* node4 = allocator.allocate(150);
    LUISA_ASSERT(node4 != nullptr, "Re-allocation failed");
    // Should reuse the freed space (first-fit)
    LUISA_ASSERT(node4->offset() == old_offset, "Expected allocation at offset {}, got {}", old_offset, node4->offset());

    // Cleanup
    allocator.free(node1);
    allocator.free(node3);
    allocator.free(node4);

    LUISA_INFO("Basic allocate/free test passed.");
}

// Test allocation with different alignments
void test_alignment() {
    LUISA_INFO("Testing alignment constraints...");

    // Test with 16-byte alignment
    FirstFit allocator(1024, 16);
    LUISA_ASSERT(allocator.alignment() == 16, "Alignment should be 16");

    auto* node1 = allocator.allocate(50);
    LUISA_ASSERT(node1 != nullptr, "Allocation failed");
    LUISA_ASSERT(node1->offset() % 16 == 0, "Offset {} should be aligned to 16", node1->offset());

    auto* node2 = allocator.allocate(60);
    LUISA_ASSERT(node2 != nullptr, "Allocation failed");
    LUISA_ASSERT(node2->offset() % 16 == 0, "Offset {} should be aligned to 16", node2->offset());

    allocator.free(node1);
    allocator.free(node2);

    // Test with 64-byte alignment (cache line)
    FirstFit allocator64(4096, 64);
    auto* node3 = allocator64.allocate(100);
    LUISA_ASSERT(node3 != nullptr, "Allocation failed");
    LUISA_ASSERT(node3->offset() % 64 == 0, "Offset {} should be aligned to 64", node3->offset());

    allocator64.free(node3);

    LUISA_INFO("Alignment test passed.");
}

// Test first-fit allocation strategy
void test_first_fit_strategy() {
    LUISA_INFO("Testing first-fit allocation strategy...");

    FirstFit allocator(1024, 8);

    // Create fragmentation pattern: alloc A, B, C, then free A and B
    auto* nodeA = allocator.allocate(100);
    auto* nodeB = allocator.allocate(100);
    auto* nodeC = allocator.allocate(100);

    size_t offsetA = nodeA->offset();
    allocator.free(nodeA);
    allocator.free(nodeB);

    // Now we have a hole at offset 0 of size ~200
    // Allocate a block of 50 - should use first-fit (offset 0)
    auto* nodeD = allocator.allocate(50);
    LUISA_ASSERT(nodeD != nullptr, "Allocation failed");
    LUISA_ASSERT(nodeD->offset() == offsetA, "First-fit should use first available block at offset {}, got {}",
                 offsetA, nodeD->offset());

    allocator.free(nodeC);
    allocator.free(nodeD);

    LUISA_INFO("First-fit strategy test passed.");
}

// Test best-fit allocation strategy
void test_best_fit_strategy() {
    LUISA_INFO("Testing best-fit allocation strategy...");

    FirstFit allocator(1024, 8);

    // Create specific fragmentation: allocate and free to create holes
    auto* node1 = allocator.allocate(200);
    auto* node2 = allocator.allocate(300);
    auto* node3 = allocator.allocate(200);

    allocator.free(node1);
    allocator.free(node3);

    // Both holes are size 200. Allocate 150 using best-fit
    auto* node4 = allocator.allocate_best_fit(150);
    LUISA_ASSERT(node4 != nullptr, "Best-fit allocation failed");
    LUISA_ASSERT(node4->size() >= 150, "Allocated size should be >= 150");

    allocator.free(node2);
    allocator.free(node4);

    LUISA_INFO("Best-fit strategy test passed.");
}

// Test memory exhaustion
void test_memory_exhaustion() {
    LUISA_INFO("Testing memory exhaustion...");

    FirstFit allocator(256, 8);

    // Allocate most of the memory
    auto* node1 = allocator.allocate(200);
    LUISA_ASSERT(node1 != nullptr, "Allocation of 200 bytes should succeed");

    // Try to allocate more than available
    auto* node2 = allocator.allocate(100);
    // Note: May or may not succeed depending on internal fragmentation
    // Just verify we don't crash

    if (node2) {
        allocator.free(node2);
    }
    allocator.free(node1);

    LUISA_INFO("Memory exhaustion test passed.");
}

// Test move semantics
void test_move_semantics() {
    LUISA_INFO("Testing move semantics...");

    // Move constructor
    {
        FirstFit allocator1(1024, 8);
        auto* node = allocator1.allocate(100);
        LUISA_ASSERT(node != nullptr, "Allocation failed");
        size_t offset = node->offset();

        // Move construct
        FirstFit allocator2(std::move(allocator1));

        // Verify allocator2 has the allocation
        LUISA_ASSERT(allocator2.size() == 1024, "Move constructor: size should be preserved");

        // Free from moved allocator
        allocator2.free(node);

        // Verify we can allocate from moved allocator
        auto* node2 = allocator2.allocate(200);
        LUISA_ASSERT(node2 != nullptr, "Allocation from moved allocator failed");
        allocator2.free(node2);
    }

    // Move assignment
    {
        FirstFit allocator1(1024, 8);
        auto* node = allocator1.allocate(100);
        LUISA_ASSERT(node != nullptr, "Allocation failed");

        FirstFit allocator2(512, 16);
        allocator2 = std::move(allocator1);

        LUISA_ASSERT(allocator2.size() == 1024, "Move assignment: size should be from source");
        LUISA_ASSERT(allocator2.alignment() == 8, "Move assignment: alignment should be from source");

        allocator2.free(node);
    }

    LUISA_INFO("Move semantics test passed.");
}

// Test dump_free_list for debugging
void test_dump_free_list() {
    LUISA_INFO("Testing dump_free_list...");

    FirstFit allocator(1024, 8);

    // Get initial free list dump
    auto free_list_str = allocator.dump_free_list();
    LUISA_ASSERT(!free_list_str.empty(), "Free list dump should not be empty");
    LUISA_INFO("Initial free list: {}", free_list_str);

    // Allocate and free to create fragmentation
    auto* node1 = allocator.allocate(100);
    auto* node2 = allocator.allocate(100);
    auto* node3 = allocator.allocate(100);

    allocator.free(node2);// Create a hole

    free_list_str = allocator.dump_free_list();
    LUISA_INFO("Free list after fragmentation: {}", free_list_str);

    allocator.free(node1);
    allocator.free(node3);

    free_list_str = allocator.dump_free_list();
    LUISA_INFO("Free list after all freed: {}", free_list_str);

    LUISA_INFO("Dump free list test passed.");
}

// Test fragmentation and coalescing
void test_fragmentation_and_coalescing() {
    LUISA_INFO("Testing fragmentation and coalescing...");

    FirstFit allocator(1024, 8);

    // Create multiple allocations
    auto* node1 = allocator.allocate(100);
    auto* node2 = allocator.allocate(100);
    auto* node3 = allocator.allocate(100);
    auto* node4 = allocator.allocate(100);

    size_t offset1 = node1->offset();
    size_t offset2 = node2->offset();
    size_t offset3 = node3->offset();
    size_t offset4 = node4->offset();

    // Free in reverse order to test coalescing
    allocator.free(node4);
    allocator.free(node3);
    allocator.free(node2);
    allocator.free(node1);

    // After freeing all, we should be able to allocate the full size again
    auto* big_node = allocator.allocate(400);
    LUISA_ASSERT(big_node != nullptr, "Should be able to allocate 400 bytes after coalescing");
    LUISA_ASSERT(big_node->offset() == offset1, "Large allocation should start at beginning");

    allocator.free(big_node);

    LUISA_INFO("Fragmentation and coalescing test passed.");
}

// Test exact fit allocation
void test_exact_fit() {
    LUISA_INFO("Testing exact fit allocation...");

    FirstFit allocator(1024, 8);

    // Allocate a specific size
    auto* node1 = allocator.allocate(256);
    LUISA_ASSERT(node1 != nullptr, "Allocation failed");
    size_t allocated_size = node1->size();

    allocator.free(node1);

    // Allocate the exact same size again
    auto* node2 = allocator.allocate(allocated_size);
    LUISA_ASSERT(node2 != nullptr, "Exact fit allocation failed");

    allocator.free(node2);

    LUISA_INFO("Exact fit test passed.");
}

// Test multiple allocations/deallocations pattern
void test_multiple_allocs_pattern() {
    LUISA_INFO("Testing multiple allocations pattern...");

    FirstFit allocator(4096, 16);
    std::vector<FirstFit::Node*> nodes;

    // Allocate many small blocks
    for (int i = 0; i < 20; ++i) {
        auto* node = allocator.allocate(64);
        LUISA_ASSERT(node != nullptr, "Allocation {} failed", i);
        nodes.push_back(node);
    }

    // Free every other block (create fragmentation)
    for (size_t i = 0; i < nodes.size(); i += 2) {
        allocator.free(nodes[i]);
    }

    // Allocate blocks that should fit in freed spaces
    std::vector<FirstFit::Node*> new_nodes;
    for (int i = 0; i < 10; ++i) {
        auto* node = allocator.allocate(32);
        LUISA_ASSERT(node != nullptr, "Re-allocation {} failed", i);
        new_nodes.push_back(node);
    }

    // Cleanup remaining nodes
    for (size_t i = 1; i < nodes.size(); i += 2) {
        allocator.free(nodes[i]);
    }
    for (auto* node : new_nodes) {
        allocator.free(node);
    }

    LUISA_INFO("Multiple allocations pattern test passed.");
}

// Test edge case: allocate 0 bytes
void test_zero_allocation() {
    LUISA_INFO("Testing zero allocation edge case...");

    FirstFit allocator(1024, 8);

    // Try to allocate 0 bytes
    auto* node = allocator.allocate(0);
    // Implementation may return nullptr or a valid node
    if (node) {
        allocator.free(node);
    }

    LUISA_INFO("Zero allocation test passed.");
}

int main() {
    log_level_verbose();

    LUISA_INFO("Starting FirstFit tests...");

    test_basic_construction();
    test_basic_allocate_free();
    test_alignment();
    test_first_fit_strategy();
    test_best_fit_strategy();
    test_memory_exhaustion();
    test_move_semantics();
    test_dump_free_list();
    test_fragmentation_and_coalescing();
    test_exact_fit();
    test_multiple_allocs_pattern();
    test_zero_allocation();

    LUISA_INFO("All FirstFit tests passed!");

    return 0;
}
