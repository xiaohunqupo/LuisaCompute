find_package(hip)
find_package(hiprtc QUIET)

if (hip_FOUND)
    message(STATUS "Build with HIP backend: ${hip_VERSION}")
    set(LUISA_HIP_SOURCES
            ../common/default_binary_io.cpp ../common/default_binary_io.h
            hip_device.cpp hip_device.h
            hip_buffer.cpp hip_buffer.h
            hip_bindless_array.cpp hip_bindless_array.h
            hip_accel.cpp hip_accel.h
            hip_callback_context.h
            hip_command_encoder.cpp hip_command_encoder.h
            hip_event.cpp hip_event.h
            hip_mesh.cpp hip_mesh.h
            hip_shader.h hip_shader.cpp
            hip_shader_metadata.h
            hip_shader_native.h hip_shader_native.cpp
            hip_shader_printer.h
            hip_stage_buffer_pool.cpp hip_stage_buffer_pool.h
            hip_stream.cpp hip_stream.h
            hip_swapchain.cpp hip_swapchain.h
            hip_texture.cpp hip_texture.h
    )
    luisa_compute_add_backend(hip SOURCES ${LUISA_HIP_SOURCES})

# Add LLVM codegen for HIP
add_subdirectory(llvm_codegen)

target_link_libraries(luisa-compute-backend-hip PRIVATE
        luisa-compute-ast
        luisa-compute-runtime
        hip::host
)
    if (TARGET hiprtc::hiprtc)
        target_link_libraries(luisa-compute-backend-hip PRIVATE hiprtc::hiprtc)
    elseif (TARGET hip::hiprtc)
        target_link_libraries(luisa-compute-backend-hip PRIVATE hip::hiprtc)
    else ()
        message(WARNING "hiprtc target not found; linking against libhiprtc.so by name")
        target_link_libraries(luisa-compute-backend-hip PRIVATE hiprtc)
    endif ()
    if (TARGET luisa-compute-vulkan-swapchain)
        target_link_libraries(luisa-compute-backend-hip PRIVATE luisa-compute-vulkan-swapchain)
    endif ()
    if (LUISA_COMPUTE_HIP_PLATFORM_NVIDIA)
        find_package(CUDAToolkit REQUIRED)
        target_compile_definitions(luisa-compute-backend-hip PRIVATE __HIP_PLATFORM_NVIDIA__=1)
        if (TARGET CUDA::cudart_static)
            target_link_libraries(luisa-compute-backend-hip PRIVATE CUDA::cudart_static)
        else ()
            target_link_libraries(luisa-compute-backend-hip PRIVATE CUDA::cudart)
        endif ()
    endif ()

    # HIPRT integration: build from source (submodule at src/ext/HIPRT)
    # Uses system hipcc for compilation
    set(HIPRT_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/ext/HIPRT")
    include(${CMAKE_SOURCE_DIR}/src/ext/hiprt_from_source.cmake)

    # Link against the from-source HIPRT target
    target_include_directories(luisa-compute-backend-hip PRIVATE
            ${HIPRT_FROM_SOURCE_INCLUDE_DIR}
            ${HIPRT_SOURCE_DIR})
    target_link_libraries(luisa-compute-backend-hip PRIVATE hiprt_from_source)

    set(LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR ${HIPRT_FROM_SOURCE_INCLUDE_DIR})

    # compile HIPRT wrapper .hip -> .bc at build time
    set(LUISA_HIP_RT_WRAPPER_SOURCE ${CMAKE_CURRENT_LIST_DIR}/hip_rt_device_wrapper.hip)
    set(LUISA_HIP_RT_WRAPPER_BC ${CMAKE_CURRENT_LIST_DIR}/hip_rt_wrapper.bc)
    # Include both the generated header dir and the source tree
    set(_HIPRT_WRAPPER_INCLUDE_ARGS -I ${LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR} -I ${HIPRT_SOURCE_DIR})
    add_custom_target(luisa-compute-backend-hip-compile-wrapper
            COMMAND $<TARGET_FILE:luisa_hip_compile_wrapper>
                -i ${LUISA_HIP_RT_WRAPPER_SOURCE}
                -o ${LUISA_HIP_RT_WRAPPER_BC}
                ${_HIPRT_WRAPPER_INCLUDE_ARGS}
            BYPRODUCTS ${LUISA_HIP_RT_WRAPPER_BC}
            DEPENDS luisa_hip_compile_wrapper ${LUISA_HIP_RT_WRAPPER_SOURCE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            COMMENT "Compiling HIPRT wrapper to LLVM bitcode..."
            VERBATIM)

    # embed .bc -> .cpp/.h via luisa_embed_device_lib
    add_custom_target(luisa-compute-backend-hip-embed-wrapper
            COMMAND $<TARGET_FILE:luisa_embed_device_lib>
                ${LUISA_HIP_RT_WRAPPER_BC}
                --unsigned --prefix luisa_compute_hip_
                -o hip_rt_wrapper_bitcode_embedded.cpp
                -h hip_rt_wrapper_bitcode_embedded.h
            BYPRODUCTS
                ${CMAKE_CURRENT_LIST_DIR}/hip_rt_wrapper_bitcode_embedded.cpp
                ${CMAKE_CURRENT_LIST_DIR}/hip_rt_wrapper_bitcode_embedded.h
            DEPENDS luisa_embed_device_lib luisa-compute-backend-hip-compile-wrapper
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            COMMENT "Embedding HIPRT wrapper bitcode..."
            VERBATIM)

    target_sources(luisa-compute-backend-hip PRIVATE
            hip_rt_wrapper_bitcode_embedded.cpp
            hip_rt_wrapper_bitcode_embedded.h)
    add_dependencies(luisa-compute-backend-hip luisa-compute-backend-hip-embed-wrapper)

    # copy HIPRT runtime files to binary output directory
    # Copy the from-source built shared library and bitcode artifacts
    add_custom_command(TARGET luisa-compute-backend-hip POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:hiprt_from_source>
                $<TARGET_FILE_DIR:luisa-compute-backend-hip>)
    # Copy bitcode/hipfb artifacts needed at runtime
    add_custom_command(TARGET luisa-compute-backend-hip POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${HIPRT_FROM_SOURCE_BITCODE_DIR}/${HIPRT_FROM_SOURCE_LIB_NAME}_${HIP_VERSION_STR}_amd_lib_linux.bc"
                "${HIPRT_FROM_SOURCE_BITCODE_DIR}/${HIPRT_FROM_SOURCE_LIB_NAME}_${HIP_VERSION_STR}_amd.hipfb"
                "${HIPRT_FROM_SOURCE_BITCODE_DIR}/oro_compiled_kernels.hipfb"
                $<TARGET_FILE_DIR:luisa-compute-backend-hip>)

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "HIP not found. The HIP backend will not be built.")
else ()
    message(WARNING "HIP not found. The HIP backend will not be built.")
endif ()
