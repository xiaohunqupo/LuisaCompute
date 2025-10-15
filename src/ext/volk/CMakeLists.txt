add_library(luisa-compute-ext-volk STATIC volk.c)
target_compile_definitions(luisa-compute-ext-volk PUBLIC VK_NO_PROTOTYPES)
target_include_directories(luisa-compute-ext-volk PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
luisa_compute_install_extension(luisa-compute-ext-volk INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(luisa-compute-ext-volk PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME luisa-ext-volk)
