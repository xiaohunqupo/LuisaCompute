find_package(hip)

if (hip_FOUND)
    message(STATUS "Build with HIP backend: ${hip_VERSION}")
    set(LUISA_HIP_SOURCES
            ../common/default_binary_io.cpp ../common/default_binary_io.h
            hip_device.cpp hip_device.h
            hip_buffer.cpp hip_buffer.h
    )
    luisa_compute_add_backend(hip SOURCES ${LUISA_HIP_SOURCES})
    target_link_libraries(luisa-compute-backend-hip PRIVATE
            luisa-compute-ast
            luisa-compute-runtime
            hip::host)
elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "HIP not found. The HIP backend will not be built.")
else ()
    message(WARNING "HIP not found. The HIP backend will not be built.")
endif ()
