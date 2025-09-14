if (LUISA_COMPUTE_ENABLE_DX OR LUISA_COMPUTE_ENABLE_VULKAN)

    file(GLOB_RECURSE LUISA_COMPUTE_HLSL_SOURCES CONFIGURE_DEPENDS *.cpp *.h)

    add_library(luisa-compute-hlsl-codegen STATIC ${LUISA_COMPUTE_HLSL_SOURCES})
    target_precompile_headers(luisa-compute-hlsl-codegen PRIVATE lc_hlsl_pch.h)
    target_link_libraries(luisa-compute-hlsl-codegen PUBLIC luisa-compute-runtime luisa-compute-core luisa-compute-vstl)

    set(ZLIB_USE_STATIC_LIBS ON)
    find_package(ZLIB)
    if (ZLIB_FOUND)
        message(STATUS "Found ZLIB: ${ZLIB_LIBRARIES}")
        target_link_libraries(luisa-compute-hlsl-codegen PRIVATE ZLIB::ZLIB)
    else () # use maxwell's favorite xrepo otherwise
        target_include_directories(luisa-compute-hlsl-codegen PRIVATE "../../../../xrepo/packages/z/zlib/port/zlib/include")
        # convert zlib path to absolute path
        set(ZLIB_LIB_FILE "../../../../xrepo/packages/z/zlib/port/zlib/lib/windows/x64/zlibstatic.lib")
        cmake_path(ABSOLUTE_PATH ZLIB_LIB_FILE)
        target_link_libraries(luisa-compute-hlsl-codegen PRIVATE ${ZLIB_LIB_FILE})
    endif ()

endif ()
