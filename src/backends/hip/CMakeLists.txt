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
            hip_mesh.cpp hip_mesh.h
            hip_stage_buffer_pool.cpp hip_stage_buffer_pool.h
            hip_stream.cpp hip_stream.h
            hip_swapchain.cpp hip_swapchain.h
            hip_texture.cpp hip_texture.h
    )
    luisa_compute_add_backend(hip SOURCES ${LUISA_HIP_SOURCES})
    target_link_libraries(luisa-compute-backend-hip PRIVATE
            luisa-compute-ast
            luisa-compute-runtime
            hip::host
    )
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

    # download HIPRT SDK
    set(LUISA_COMPUTE_HIPRT_SDK_VERSION "3.0.9ba63f3")
    set(LUISA_COMPUTE_HIPRT_SDK_DOWNLOAD_URL "https://gpuopen.com/download/hiprtSdk-${LUISA_COMPUTE_HIPRT_SDK_VERSION}.zip")
    set(LUISA_COMPUTE_HIPRT_SDK_SHA256 "dc1b8bd4842c74d580561449a3342911f7617e1f51ea6541fc3290e3f2e36e89")
    message(STATUS "Downloading HIPRT SDK version ${LUISA_COMPUTE_HIPRT_SDK_VERSION}...")
    include(FetchContent)
    FetchContent_Declare(
            hiprt_sdk
            URL ${LUISA_COMPUTE_HIPRT_SDK_DOWNLOAD_URL}
            URL_HASH SHA256=${LUISA_COMPUTE_HIPRT_SDK_SHA256}
            DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
    FetchContent_MakeAvailable(hiprt_sdk)
    FetchContent_GetProperties(hiprt_sdk)
    set(LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR ${hiprt_sdk_SOURCE_DIR})
    if (LINUX)
        set(LUISA_COMPUTE_HIPRT_SDK_LIB_DIR ${hiprt_sdk_SOURCE_DIR}/hiprt/linux64)
    elseif (WIN32)
        set(LUISA_COMPUTE_HIPRT_SDK_LIB_DIR ${hiprt_sdk_SOURCE_DIR}/hiprt/win)
    endif ()
    file(GLOB LUISA_COMPUTE_HIPRT_SDK_LIBS
            "${LUISA_COMPUTE_HIPRT_SDK_LIB_DIR}/hiprt*.lib"
            "${LUISA_COMPUTE_HIPRT_SDK_LIB_DIR}/libhiprt*.so")
    message(STATUS "HIPRT SDK include directory: ${LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR}")
    message(STATUS "HIPRT SDK library directory: ${LUISA_COMPUTE_HIPRT_SDK_LIB_DIR}")
    message(STATUS "HIPRT SDK link libraries: ${LUISA_COMPUTE_HIPRT_SDK_LIBS}")

    # link HIPRT SDK
    target_include_directories(luisa-compute-backend-hip PRIVATE ${LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR})
    target_link_libraries(luisa-compute-backend-hip PRIVATE ${LUISA_COMPUTE_HIPRT_SDK_LIBS})

    # copy HIPRT libraries to binary output directory
    add_custom_command(TARGET luisa-compute-backend-hip POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${LUISA_COMPUTE_HIPRT_SDK_LIB_DIR}
            $<TARGET_FILE_DIR:luisa-compute-backend-hip>
            DEPENDS ${LUISA_COMPUTE_HIPRT_SDK_LIBS})

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "HIP not found. The HIP backend will not be built.")
else ()
    message(WARNING "HIP not found. The HIP backend will not be built.")
endif ()
