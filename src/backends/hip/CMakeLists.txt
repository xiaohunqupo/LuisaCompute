find_package(hip)

if (hip_FOUND)
    message(STATUS "Build with HIP backend: ${hip_VERSION}")
    set(LUISA_HIP_SOURCES
            ../common/default_binary_io.cpp ../common/default_binary_io.h
            hip_device.cpp hip_device.h
            hip_buffer.cpp hip_buffer.h
            hip_callback_context.h
            hip_command_encoder.cpp hip_command_encoder.h
            hip_event.cpp hip_event.h
            hip_stage_buffer_pool.cpp hip_stage_buffer_pool.h
            hip_stream.cpp hip_stream.h
            hip_swapchain.cpp hip_swapchain.h
            hip_texture.cpp hip_texture.h
    )
    luisa_compute_add_backend(hip SOURCES ${LUISA_HIP_SOURCES})
    target_link_libraries(luisa-compute-backend-hip PRIVATE
            luisa-compute-ast
            luisa-compute-runtime
            luisa-compute-vulkan-swapchain
            hip::host)
    if (LUISA_COMPUTE_HIP_PLATFORM_NVIDIA)
        find_package(CUDAToolkit REQUIRED)
        target_compile_definitions(luisa-compute-backend-hip PRIVATE __HIP_PLATFORM_NVIDIA__=1)
        if (TARGET CUDA::cudart_static)
            target_link_libraries(luisa-compute-backend-hip PRIVATE CUDA::cudart_static)
        else ()
            target_link_libraries(luisa-compute-backend-hip PRIVATE CUDA::cudart)
        endif ()
    endif ()
elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "HIP not found. The HIP backend will not be built.")
else ()
    message(WARNING "HIP not found. The HIP backend will not be built.")
endif ()
