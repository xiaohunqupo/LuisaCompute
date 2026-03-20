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

    # Use system-installed HIPRT SDK from ROCm (preferred for matching ROCm version).
    # Falls back to downloading HIPRT 3.0 SDK if system HIPRT is not found.
    set(LUISA_COMPUTE_USE_SYSTEM_HIPRT OFF)

    # Check for system HIPRT (installed with ROCm, e.g. /opt/rocm/include/hiprt)
    find_path(LUISA_COMPUTE_HIPRT_SYSTEM_INCLUDE_DIR
            NAMES hiprt/hiprt.h
            PATHS /opt/rocm/include
            NO_DEFAULT_PATH)
    if (LUISA_COMPUTE_HIPRT_SYSTEM_INCLUDE_DIR)
        # System HIPRT headers found; now find the library
        find_library(LUISA_COMPUTE_HIPRT_SYSTEM_LIB
                NAMES hiprt0300164 hiprt64
                PATHS /opt/rocm/lib
                NO_DEFAULT_PATH)
        if (LUISA_COMPUTE_HIPRT_SYSTEM_LIB)
            set(LUISA_COMPUTE_USE_SYSTEM_HIPRT ON)
            # Include dir is parent of hiprt/ so headers are included as <hiprt/hiprt.h>
            set(LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR ${LUISA_COMPUTE_HIPRT_SYSTEM_INCLUDE_DIR})
            set(LUISA_COMPUTE_HIPRT_SDK_LIB_DIR /opt/rocm/lib)
            set(LUISA_COMPUTE_HIPRT_SDK_LIBS ${LUISA_COMPUTE_HIPRT_SYSTEM_LIB})
            message(STATUS "Using system HIPRT SDK from ROCm:")
            message(STATUS "  HIPRT include: ${LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR}")
            message(STATUS "  HIPRT library: ${LUISA_COMPUTE_HIPRT_SDK_LIBS}")
            message(STATUS "  HIPRT lib dir: ${LUISA_COMPUTE_HIPRT_SDK_LIB_DIR}")
        endif ()
    endif ()

    if (NOT LUISA_COMPUTE_USE_SYSTEM_HIPRT)
        # Fallback: download HIPRT SDK
        set(LUISA_COMPUTE_HIPRT_SDK_VERSION "3.0.9ba63f3")
        set(LUISA_COMPUTE_HIPRT_SDK_DOWNLOAD_URL "https://gpuopen.com/download/hiprtSdk-${LUISA_COMPUTE_HIPRT_SDK_VERSION}.zip")
        set(LUISA_COMPUTE_HIPRT_SDK_SHA256 "dc1b8bd4842c74d580561449a3342911f7617e1f51ea6541fc3290e3f2e36e89")
        message(STATUS "System HIPRT not found; downloading HIPRT SDK version ${LUISA_COMPUTE_HIPRT_SDK_VERSION}...")
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
    endif ()

    # link HIPRT SDK
    target_include_directories(luisa-compute-backend-hip PRIVATE ${LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR})
    target_link_libraries(luisa-compute-backend-hip PRIVATE ${LUISA_COMPUTE_HIPRT_SDK_LIBS})

    # compile HIPRT wrapper .hip -> .bc at build time
    set(LUISA_HIP_RT_WRAPPER_SOURCE ${CMAKE_CURRENT_LIST_DIR}/hip_rt_device_wrapper.hip)
    set(LUISA_HIP_RT_WRAPPER_BC ${CMAKE_CURRENT_LIST_DIR}/hip_rt_wrapper.bc)
    add_custom_target(luisa-compute-backend-hip-compile-wrapper
            COMMAND $<TARGET_FILE:luisa_hip_compile_wrapper>
                -i ${LUISA_HIP_RT_WRAPPER_SOURCE}
                -o ${LUISA_HIP_RT_WRAPPER_BC}
                -I ${LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR}
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

    # standalone HIPRT test (bypasses LuisaCompute, tests HIPRT SDK directly)
    add_executable(test_hiprt_standalone ${CMAKE_SOURCE_DIR}/src/tests/test_hiprt_standalone.cpp)
    target_link_libraries(test_hiprt_standalone PRIVATE hip::host ${LUISA_COMPUTE_HIPRT_SDK_LIBS})
    if (TARGET hiprtc::hiprtc)
        target_link_libraries(test_hiprt_standalone PRIVATE hiprtc::hiprtc)
    elseif (TARGET hip::hiprtc)
        target_link_libraries(test_hiprt_standalone PRIVATE hip::hiprtc)
    else ()
        target_link_libraries(test_hiprt_standalone PRIVATE hiprtc)
    endif ()
    target_include_directories(test_hiprt_standalone PRIVATE ${LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR})
    target_compile_definitions(test_hiprt_standalone PRIVATE
            HIPRT_SDK_INCLUDE_DIR="${LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR}")
    set_target_properties(test_hiprt_standalone PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # copy HIPRT libraries to binary output directory
    if (LUISA_COMPUTE_USE_SYSTEM_HIPRT)
        # For system HIPRT, copy the specific runtime files needed
        file(GLOB LUISA_COMPUTE_HIPRT_RUNTIME_FILES
                "${LUISA_COMPUTE_HIPRT_SDK_LIB_DIR}/libhiprt*.so"
                "${LUISA_COMPUTE_HIPRT_SDK_LIB_DIR}/hiprt*.hipfb"
                "${LUISA_COMPUTE_HIPRT_SDK_LIB_DIR}/hiprt*_lib_linux.bc"
                "${LUISA_COMPUTE_HIPRT_SDK_LIB_DIR}/oro_compiled_kernels.hipfb")
        foreach (HIPRT_FILE ${LUISA_COMPUTE_HIPRT_RUNTIME_FILES})
            add_custom_command(TARGET luisa-compute-backend-hip POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${HIPRT_FILE}
                    $<TARGET_FILE_DIR:luisa-compute-backend-hip>)
        endforeach ()
    else ()
        add_custom_command(TARGET luisa-compute-backend-hip POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${LUISA_COMPUTE_HIPRT_SDK_LIB_DIR}
                $<TARGET_FILE_DIR:luisa-compute-backend-hip>
                DEPENDS ${LUISA_COMPUTE_HIPRT_SDK_LIBS})
    endif ()

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "HIP not found. The HIP backend will not be built.")
else ()
    message(WARNING "HIP not found. The HIP backend will not be built.")
endif ()
