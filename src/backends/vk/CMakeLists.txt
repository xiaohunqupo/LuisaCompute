if (LUISA_COMPUTE_ENABLE_VULKAN)
    message(STATUS "Build with Vulkan backend")
    file(GLOB LUISA_COMPUTE_VK_SOURCES CONFIGURE_DEPENDS *.cpp *.h)
    luisa_compute_add_backend(vk SOURCES ${LUISA_COMPUTE_VK_SOURCES})
    target_link_libraries(luisa-compute-backend-vk PUBLIC luisa-compute-ext-volk luisa-compute-hlsl-codegen)
    if (APPLE)
        enable_language(OBJC)
        enable_language(OBJCXX)
        target_sources(luisa-compute-backend-vk PRIVATE ../common/moltenvk_surface.mm)
        target_link_libraries(luisa-compute-backend-vk PRIVATE
                "-framework Foundation"
                "-framework Metal"
                "-framework QuartzCore"
                "-framework AppKit")
    endif ()
    target_precompile_headers(luisa-compute-backend-vk PRIVATE lc_vk_pch.h)
    if (WIN32)
        target_compile_definitions(luisa-compute-backend-vk PRIVATE VK_USE_PLATFORM_WIN32_KHR)
    elseif (UNIX AND NOT APPLE)
        target_compile_definitions(luisa-compute-backend-vk PRIVATE VK_USE_PLATFORM_XCB_KHR)
    endif ()
    target_compile_definitions(luisa-compute-backend-vk PRIVATE USE_SPIRV)
endif ()
