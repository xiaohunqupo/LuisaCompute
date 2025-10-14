add_library(volk STATIC volk.c)
target_compile_definitions(volk PUBLIC VK_NO_PROTOTYPES)
target_include_directories(volk PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
luisa_compute_install_extension(volk)
