if (LUISA_COMPUTE_ENABLE_DX OR LUISA_COMPUTE_ENABLE_VULKAN)

    file(GLOB_RECURSE LUISA_COMPUTE_HLSL_SOURCES CONFIGURE_DEPENDS *.cpp *.h)

    add_library(luisa-compute-hlsl-codegen STATIC ${LUISA_COMPUTE_HLSL_SOURCES})
    target_precompile_headers(luisa-compute-hlsl-codegen PRIVATE lc_hlsl_pch.h)
    target_link_libraries(luisa-compute-hlsl-codegen PUBLIC luisa-compute-runtime luisa-compute-core luisa-compute-vstl)

endif ()
