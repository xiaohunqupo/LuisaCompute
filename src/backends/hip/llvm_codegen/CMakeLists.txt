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

    # -----------------------------------------------------------------------
    # Compile HIPRT wrapper .hip -> per-arch .bc at build time.
    # Each architecture gets its own bitcode so arch-specific features
    # (e.g., hardware BVH stack on gfx1200/1201) are correctly compiled.
    # -----------------------------------------------------------------------
    set(LUISA_HIP_RT_WRAPPER_SOURCE ${CMAKE_CURRENT_LIST_DIR}/hip_rt_device_wrapper.hip)
    # Include both the generated header dir and the source tree
    set(_HIPRT_WRAPPER_INCLUDE_ARGS -I ${LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR} -I ${HIPRT_SOURCE_DIR})
    # Detect target GPU architecture from environment or use default
    if (DEFINED ENV{HIPRT_GPU_ARCH} AND NOT "$ENV{HIPRT_GPU_ARCH}" STREQUAL "")
        set(_hip_wrapper_arch "$ENV{HIPRT_GPU_ARCH}")
    elseif (DEFINED HIPRT_GPU_ARCH AND NOT "${HIPRT_GPU_ARCH}" STREQUAL "")
        set(_hip_wrapper_arch "${HIPRT_GPU_ARCH}")
    else ()
        # Default: compile for common RDNA architectures
        set(_hip_wrapper_arch "gfx1030;gfx1100;gfx1200;gfx1201")
    endif ()

    # Per-arch wrapper compilation and embedding
    set(_wrapper_bc_dir "${CMAKE_CURRENT_LIST_DIR}")
    set(_wrapper_all_bcs "")
    set(_wrapper_all_embedded_sources "")

    foreach (_arch ${_hip_wrapper_arch})
        set(_wrapper_bc "${_wrapper_bc_dir}/hip_rt_wrapper_${_arch}.bc")
        set(_wrapper_cpp "${_wrapper_bc_dir}/hip_rt_wrapper_${_arch}_embedded.cpp")
        set(_wrapper_h "${_wrapper_bc_dir}/hip_rt_wrapper_${_arch}_embedded.h")

        # Compile wrapper for this arch
        add_custom_command(
                OUTPUT "${_wrapper_bc}"
                COMMAND $<TARGET_FILE:luisa_hip_compile_wrapper>
                -i ${LUISA_HIP_RT_WRAPPER_SOURCE}
                -o "${_wrapper_bc}"
                -a "${_arch}"
                ${_HIPRT_WRAPPER_INCLUDE_ARGS}
                DEPENDS luisa_hip_compile_wrapper ${LUISA_HIP_RT_WRAPPER_SOURCE}
                        ${HIPRT_SOURCE_DIR}/hiprt/impl/hiprt_device_impl.h
                        ${HIPRT_SOURCE_DIR}/hiprt/hiprt_device.h
                        ${HIPRT_SOURCE_DIR}/hiprt/hiprt_common.h
                WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                COMMENT "[HIPRT wrapper] Compiling for ${_arch}..."
                VERBATIM)

        # Embed the per-arch bitcode into C++ source
        add_custom_command(
                OUTPUT "${_wrapper_cpp}" "${_wrapper_h}"
                COMMAND $<TARGET_FILE:luisa_embed_device_lib>
                "${_wrapper_bc}"
                --unsigned --prefix luisa_compute_hip_
                -o "${_wrapper_cpp}"
                -h "${_wrapper_h}"
                DEPENDS luisa_embed_device_lib "${_wrapper_bc}"
                WORKING_DIRECTORY "${_wrapper_bc_dir}"
                COMMENT "[HIPRT wrapper] Embedding ${_arch} bitcode..."
                VERBATIM)

        list(APPEND _wrapper_all_bcs "${_wrapper_bc}")
        list(APPEND _wrapper_all_embedded_sources "${_wrapper_cpp}" "${_wrapper_h}")
    endforeach ()

    add_custom_target(luisa-compute-backend-hip-embed-wrapper
            DEPENDS ${_wrapper_all_bcs} ${_wrapper_all_embedded_sources})
    add_dependencies(luisa-compute-backend-hip luisa-compute-backend-hip-embed-wrapper)
    target_sources(luisa-compute-backend-hip PRIVATE ${_wrapper_all_embedded_sources})

    # -----------------------------------------------------------------------
    # Extract per-arch HIPRT library bitcode from the bundled .bc and embed.
    # This allows our LLVM codegen to link the HIPRT traversal functions
    # directly into the kernel module for full LTO inlining, bypassing
    # HIPRT's orortcLinkComplete which does not optimize across modules.
    # -----------------------------------------------------------------------
    find_program(CLANG_OFFLOAD_BUNDLER clang-offload-bundler
            HINTS /opt/rocm/llvm/bin $ENV{ROCM_PATH}/llvm/bin /usr/bin)
    if (NOT CLANG_OFFLOAD_BUNDLER)
        message(FATAL_ERROR "clang-offload-bundler not found; required for HIPRT LTO")
    endif ()

    set(_hiprt_bundle_bc "${HIPRT_FROM_SOURCE_BITCODE_DIR}/${HIPRT_FROM_SOURCE_LIB_NAME}_${HIP_VERSION_STR}_amd_lib_linux.bc")
    set(_hiprt_lib_bc_dir "${CMAKE_CURRENT_LIST_DIR}")

    # Collect per-arch extracted bitcode files and embedded source files
    set(_hiprt_lib_extracted_bcs "")
    set(_hiprt_lib_embedded_sources "")
    set(_hiprt_lib_embed_targets "")

    foreach (_arch ${_hip_wrapper_arch})
        set(_extracted_bc "${_hiprt_lib_bc_dir}/hiprt_library_${_arch}.bc")
        set(_embedded_cpp "${_hiprt_lib_bc_dir}/hiprt_library_${_arch}_embedded.cpp")
        set(_embedded_h   "${_hiprt_lib_bc_dir}/hiprt_library_${_arch}_embedded.h")

        # Extract single-arch bitcode from the HIPRT bundle
        add_custom_command(
                OUTPUT "${_extracted_bc}"
                COMMAND ${CLANG_OFFLOAD_BUNDLER}
                --unbundle
                "--input=${_hiprt_bundle_bc}"
                "--output=${_extracted_bc}"
                --type=bc
                "--targets=hip-amdgcn-amd-amdhsa--${_arch}"
                DEPENDS hiprt_compile_bitcodes "${_hiprt_bundle_bc}"
                COMMENT "[HIPRT LTO] Extracting ${_arch} bitcode from HIPRT bundle..."
                VERBATIM)

        # Embed the extracted bitcode into C++ source
        add_custom_command(
                OUTPUT "${_embedded_cpp}" "${_embedded_h}"
                COMMAND $<TARGET_FILE:luisa_embed_device_lib>
                "${_extracted_bc}"
                --unsigned --prefix luisa_compute_hip_
                -o "${_embedded_cpp}"
                -h "${_embedded_h}"
                DEPENDS luisa_embed_device_lib "${_extracted_bc}"
                WORKING_DIRECTORY "${_hiprt_lib_bc_dir}"
                COMMENT "[HIPRT LTO] Embedding ${_arch} HIPRT library bitcode..."
                VERBATIM)

        list(APPEND _hiprt_lib_extracted_bcs "${_extracted_bc}")
        list(APPEND _hiprt_lib_embedded_sources "${_embedded_cpp}" "${_embedded_h}")
    endforeach ()

    add_custom_target(luisa-compute-backend-hip-embed-hiprt-lib
            DEPENDS ${_hiprt_lib_extracted_bcs} ${_hiprt_lib_embedded_sources})
    add_dependencies(luisa-compute-backend-hip luisa-compute-backend-hip-embed-hiprt-lib)
    target_sources(luisa-compute-backend-hip PRIVATE ${_hiprt_lib_embedded_sources})

endif ()
