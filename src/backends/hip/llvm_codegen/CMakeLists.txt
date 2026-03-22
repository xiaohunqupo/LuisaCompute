luisa_compute_link_llvm_into_backend(hip
        COMPONENTS core ir amdgpucodegen support irreader bitreader passes analysis linker bitwriter asmparser
        DEFINITIONS LUISA_COMPUTE_ENABLE_LLVM=1)
if (LUISA_COMPUTE_HIP_LINKED_LLVM)
    message(STATUS "Linking LLVM AMDGPU into the HIP backend")
    # find AMD OCML device library bitcode
    set(HIP_OCML_SEARCH_PATHS
            /opt/rocm/amdgcn/bitcode
            $ENV{ROCM_PATH}/amdgcn/bitcode
    )
    find_file(HIP_OCML_FILE NAMES ocml.bc HINTS ${HIP_OCML_SEARCH_PATHS} NO_DEFAULT_PATH)
    if (HIP_OCML_FILE)
        message(STATUS "Found AMD OCML: ${HIP_OCML_FILE}")
    else ()
        message(FATAL_ERROR "Could not find AMD OCML (ocml.bc) in ${HIP_OCML_SEARCH_PATHS}")
    endif ()
    # use luisa-embed-device-lib to embed OCML bitcode
    add_custom_target(luisa-compute-backend-hip-embed-ocml
            COMMAND $<TARGET_FILE:luisa_embed_device_lib> ${HIP_OCML_FILE} --unsigned --prefix luisa_compute_hip_ -o hip_codegen_llvm_device_bitcode.cpp -h hip_codegen_llvm_device_bitcode.h
            BYPRODUCTS ${CMAKE_CURRENT_LIST_DIR}/hip_codegen_llvm_device_bitcode.cpp ${CMAKE_CURRENT_LIST_DIR}/hip_codegen_llvm_device_bitcode.h
            DEPENDS luisa_embed_device_lib ${CMAKE_CURRENT_LIST_FILE}
            SOURCES ${HIP_OCML_FILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            COMMENT "Embedding AMD OCML bitcode..."
            VERBATIM)
    add_dependencies(luisa-compute-backend-hip luisa-compute-backend-hip-embed-ocml)
    target_sources(luisa-compute-backend-hip PRIVATE
            hip_codegen_llvm_device_bitcode.cpp hip_codegen_llvm_device_bitcode.h
            hip_codegen_llvm.h
            hip_codegen_llvm.cpp
            hip_codegen_llvm_impl.h
            hip_codegen_llvm_impl.cpp
            hip_codegen_llvm_impl_type.cpp
            hip_codegen_llvm_impl_func.cpp
            hip_codegen_llvm_impl_inst.cpp
            hip_codegen_llvm_impl_cflow.cpp
            hip_codegen_llvm_impl_arith.cpp
            hip_codegen_llvm_impl_var.cpp
            hip_codegen_llvm_impl_resource.cpp
            hip_codegen_llvm_impl_cast.cpp
            hip_codegen_llvm_impl_atomic.cpp
            hip_codegen_llvm_impl_sreg.cpp
            hip_codegen_llvm_impl_const.cpp
            hip_codegen_llvm_impl_debug.cpp
            hip_codegen_llvm_impl_phi.cpp
            hip_codegen_llvm_impl_cta.cpp
    )

    # compile HIPRT wrapper .hip -> .bc at build time
    set(LUISA_HIP_RT_WRAPPER_SOURCE ${CMAKE_CURRENT_LIST_DIR}/hip_rt_device_wrapper.hip)
    set(LUISA_HIP_RT_WRAPPER_BC ${CMAKE_CURRENT_LIST_DIR}/hip_rt_wrapper.bc)
    # Include both the generated header dir and the source tree
    set(_HIPRT_WRAPPER_INCLUDE_ARGS -I ${LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR} -I ${HIPRT_SOURCE_DIR})
    # Detect target GPU architecture from environment or use default
    if (DEFINED ENV{HIPRT_GPU_ARCH} AND NOT "$ENV{HIPRT_GPU_ARCH}" STREQUAL "")
        set(_hip_wrapper_arch "$ENV{HIPRT_GPU_ARCH}")
    elseif (DEFINED HIPRT_GPU_ARCH AND NOT "${HIPRT_GPU_ARCH}" STREQUAL "")
        set(_hip_wrapper_arch "${HIPRT_GPU_ARCH}")
    else ()
        # Default: compile for common RDNA architectures
        # This enables RTIP auto-detection via __gfxNNNN__ macros
        set(_hip_wrapper_arch "gfx1030;gfx1100;gfx1200;gfx1201")
    endif ()
    # Build arch flags
    set(_hip_arch_flags "")
    foreach (_arch ${_hip_wrapper_arch})
        list(APPEND _hip_arch_flags "-a" "${_arch}")
    endforeach ()
    add_custom_target(luisa-compute-backend-hip-compile-wrapper
            COMMAND $<TARGET_FILE:luisa_hip_compile_wrapper>
            -i ${LUISA_HIP_RT_WRAPPER_SOURCE}
            -o ${LUISA_HIP_RT_WRAPPER_BC}
            ${_hip_arch_flags}
            ${_HIPRT_WRAPPER_INCLUDE_ARGS}
            BYPRODUCTS ${LUISA_HIP_RT_WRAPPER_BC}
            DEPENDS luisa_hip_compile_wrapper ${LUISA_HIP_RT_WRAPPER_SOURCE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            COMMENT "Compiling HIPRT wrapper to LLVM bitcode for arch ${_hip_wrapper_arch}..."
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

endif ()
