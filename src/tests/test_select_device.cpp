// Device selection test demonstrating how to enumerate and select
// specific hardware devices by name (GeForce, Radeon RX, Arc).

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/core/logging.h>

using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    
    // Check command line arguments
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    
    // Get hardware device names for the specified backend
    luisa::vector<luisa::string> device_names = context.backend_device_names(argv[1]);
    if (device_names.empty()) {
        LUISA_WARNING("No hardware device found.");
        exit(1);
    }
    
    // Print all available devices
    size_t device_index = 0;
    for(auto&& i : device_names){
        LUISA_INFO("Found hardware device: {}", i);
    }
    
    // Find device with "GeForce" or "Radeon RX" or "Arc"
    for (size_t i = 0; i < device_names.size(); ++i) {
        luisa::string& device_name = device_names[i];
        if (device_name.find("GeForce") != luisa::string::npos ||
            device_name.find("Radeon RX") != luisa::string::npos ||
            device_name.find("Arc") != luisa::string::npos) {
            LUISA_INFO("Select device: {}", device_name);
            device_index = i;
        }
    }
    
    // Create device with the selected index
    DeviceConfig device_config{
        .device_index = device_index};
    Device device = context.create_device(argv[1], &device_config);
}
