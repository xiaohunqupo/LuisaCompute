// Test for AST (Abstract Syntax Tree) type system and attribute handling.
// This test verifies the creation of buffer types with custom attributes.

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    // Enable verbose logging for debugging
    luisa::log_level_verbose();
    
    // Initialize the compute context with the executable path
    Context context{argv[0]};
    
    // Check if backend argument is provided
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    
    // Create a compute device with the specified backend
    Device device = context.create_device(argv[1]);
    
    // Create a list of custom attributes for the buffer type
    luisa::vector<Attribute> attris;
    attris.emplace_back("attr0", "attr1");
    
    // Create a buffer type with float elements and custom attributes
    auto t = Type::buffer(Type::of<float>(), attris);
    
    // Print the type description to verify attribute handling
    LUISA_INFO("{}", t->description());
}
