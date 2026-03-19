luisa_compute_link_llvm_into_backend(hip
        COMPONENTS core ir amdgpucodegen support irreader bitreader passes analysis linker bitwriter asmparser
        DEFINITIONS LUISA_COMPUTE_ENABLE_LLVM=1)
if (LUISA_COMPUTE_HIP_LINKED_LLVM)
    message(STATUS "Linking LLVM AMDGPU into the HIP backend")
    target_sources(luisa-compute-backend-hip PRIVATE
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
endif ()
