# =============================================================================
# LuisaCompute Tests
# Organized into: unit tests, integration tests
# Uses Boost.UT (single-header) for test framework
# =============================================================================

# --- CTest integration ---
enable_testing()

# --- Shared test_main binary (for config.h-based tests) ---
# Files using LUISA_TEST_CASE_WITH_DEVICE link into this binary.
set(test_main_sources
        test_main.cpp
        config.h
        cornell_box.h
        tiny_obj_loader.h
        tinyexr.h
        test_math_util.h
        test_math_util.cpp)

luisa_compute_add_executable(test_main ${test_main_sources})
target_include_directories(test_main PRIVATE ./)

# Helper: add a source file to the shared test_main binary
function(luisa_compute_add_shared_test file)
    target_sources(test_main PRIVATE ${file})
endfunction()

# Helper: add a standalone test executable, include path, AND register with CTest.
# Usage: luisa_compute_add_test(name source LABELS label1;label2 [ARGS arg1 ...])
# Tests without LABELS compile but are NOT registered with CTest.
function(luisa_compute_add_test name source)
    cmake_parse_arguments(PARSE_ARGV 2 CT "" "" "LABELS;ARGS")
    luisa_compute_add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE ./)
    if (CT_LABELS)
        add_test(NAME ${name} COMMAND ${name} ${CT_ARGS})
        set_tests_properties(${name} PROPERTIES LABELS "${CT_LABELS}")
    endif ()
endfunction()

# =============================================================================
# Unit Tests
# =============================================================================

# --- unit/core: standalone tests (no backend needed, auto-run via CTest) ---
luisa_compute_add_test(test_basic_traits       unit/core/test_basic_traits.cpp       LABELS "unit;unit_core")
luisa_compute_add_test(test_basic_types        unit/core/test_basic_types.cpp        LABELS "unit;unit_core")
luisa_compute_add_test(test_binary_file_stream unit/core/test_binary_file_stream.cpp LABELS "unit;unit_core")
luisa_compute_add_test(test_binary_io          unit/core/test_binary_io.cpp          LABELS "unit;unit_core")
luisa_compute_add_test(test_clock              unit/core/test_clock.cpp              LABELS "unit;unit_core")
luisa_compute_add_test(test_dynamic_module     unit/core/test_dynamic_module.cpp     LABELS "unit;unit_core")
luisa_compute_add_test(test_first_fit          unit/core/test_first_fit.cpp          LABELS "unit;unit_core")
luisa_compute_add_test(test_logging            unit/core/test_logging.cpp            LABELS "unit;unit_core")
luisa_compute_add_test(test_mathematics        unit/core/test_mathematics.cpp        LABELS "unit;unit_core")
luisa_compute_add_test(test_matrix             unit/core/test_matrix.cpp             LABELS "unit;unit_core")
luisa_compute_add_test(test_pool               unit/core/test_pool.cpp               LABELS "unit;unit_core")
luisa_compute_add_test(test_type               unit/core/test_type.cpp               LABELS "unit;unit_core")

if (LUISA_COMPUTE_ENABLE_GUI)
endif ()

# --- unit/ast: standalone tests (need backend, NOT auto-run via CTest) ---
luisa_compute_add_test(test_ast             unit/ast/test_ast.cpp)
luisa_compute_add_test(test_builtin_kernel  unit/ast/test_builtin_kernel.cpp)
target_include_directories(test_builtin_kernel PRIVATE ${CMAKE_SOURCE_DIR}/src/runtime)
luisa_compute_add_test(test_manual_ast      unit/ast/test_manual_ast.cpp)

# --- unit/ast: shared test_main tests ---
luisa_compute_add_shared_test(unit/ast/test_ast_basic.cpp)

# --- unit/dsl: standalone tests (need backend, NOT auto-run via CTest) ---
luisa_compute_add_test(test_binding_group          unit/dsl/test_binding_group.cpp)
luisa_compute_add_test(test_binding_group_template unit/dsl/test_binding_group_template.cpp)
luisa_compute_add_test(test_callable               unit/dsl/test_callable.cpp)
luisa_compute_add_test(test_constant               unit/dsl/test_constant.cpp)
luisa_compute_add_test(test_dsl                    unit/dsl/test_dsl.cpp)
luisa_compute_add_test(test_dsl_sugar              unit/dsl/test_dsl_sugar.cpp)
luisa_compute_add_test(test_dsl_multithread        unit/dsl/test_dsl_multithread.cpp)
luisa_compute_add_test(test_soa                    unit/dsl/test_soa.cpp)
luisa_compute_add_test(test_soa_subview            unit/dsl/test_soa_subview.cpp)
luisa_compute_add_test(test_soa_simple             unit/dsl/test_soa_simple.cpp)
luisa_compute_add_test(test_device_math            unit/dsl/test_device_math.cpp)
luisa_compute_add_test(test_nested_callable         unit/dsl/test_nested_callable.cpp)
luisa_compute_add_test(test_normal_encoding        unit/dsl/test_normal_encoding.cpp)

# --- unit/dsl: shared test_main tests ---
luisa_compute_add_shared_test(unit/dsl/test_calc.cpp)
luisa_compute_add_shared_test(unit/dsl/test_matrix.cpp)
luisa_compute_add_shared_test(unit/dsl/test_var.cpp)

# --- unit/runtime: standalone tests (need backend, NOT auto-run via CTest) ---
luisa_compute_add_test(test_atomic              unit/runtime/test_atomic.cpp)
luisa_compute_add_test(test_atomic_queue        unit/runtime/test_atomic_queue.cpp)
luisa_compute_add_test(test_byte_buffer         unit/runtime/test_byte_buffer.cpp)
luisa_compute_add_test(test_context             unit/runtime/test_context.cpp)
luisa_compute_add_test(test_copy                unit/runtime/test_copy.cpp)
luisa_compute_add_test(test_cpu_callable        unit/runtime/test_cpu_callable.cpp)
luisa_compute_add_test(test_decoupled_look_back unit/runtime/test_decoupled_look_back.cpp)
luisa_compute_add_test(test_matrix_multiply     unit/runtime/test_matrix_multiply.cpp)
luisa_compute_add_test(test_mipmap              unit/runtime/test_mipmap.cpp)
luisa_compute_add_test(test_pinned_mem          unit/runtime/test_pinned_mem.cpp)
luisa_compute_add_test(test_printer             unit/runtime/test_printer.cpp)
luisa_compute_add_test(test_printer_custom_callback unit/runtime/test_printer_custom_callback.cpp)
luisa_compute_add_test(test_sampler             unit/runtime/test_sampler.cpp)
luisa_compute_add_test(test_shared_memory       unit/runtime/test_shared_memory.cpp)
luisa_compute_add_test(test_softmax             unit/runtime/test_softmax.cpp)
if (LUISA_COMPUTE_ENABLE_TENSOR)
    luisa_compute_add_test(test_tensor              unit/runtime/test_tensor.cpp)
endif ()
luisa_compute_add_test(test_texture_compress    unit/runtime/test_texture_compress.cpp)
luisa_compute_add_test(test_texture_io          unit/runtime/test_texture_io.cpp)
luisa_compute_add_test(test_warp                unit/runtime/test_warp.cpp)
luisa_compute_add_test(test_warp_prefix_scan    unit/runtime/test_warp_prefix_scan.cpp)

# --- unit/runtime: shared test_main tests ---
luisa_compute_add_shared_test(unit/runtime/test_buffer_io.cpp)
luisa_compute_add_shared_test(unit/runtime/test_buffer.cpp)
luisa_compute_add_shared_test(unit/runtime/test_buffer_view.cpp)
luisa_compute_add_shared_test(unit/runtime/test_device.cpp)
luisa_compute_add_shared_test(unit/runtime/test_external_buffer.cpp)
luisa_compute_add_shared_test(unit/runtime/test_gemm.cpp)
luisa_compute_add_shared_test(unit/runtime/test_shared_mem.cpp)

# --- unit/xir: standalone tests (no backend needed, auto-run via CTest) ---
luisa_compute_add_test(test_ast_to_xir unit/xir/test_ast_to_xir.cpp LABELS "unit;unit_xir")

# --- test_main: shared binary (needs backend, NOT auto-run via CTest) ---
# test_main contains shared tests that require a backend argument.
# Run manually: ./test_main <backend>

# =============================================================================
# Integration Tests (require GPU backends, NOT auto-run via CTest)
# Run manually: ./<test_name> <backend>
# =============================================================================

# --- integration/runtime: standalone tests ---
luisa_compute_add_test(test_device_debugger         integration/runtime/test_device_debugger.cpp)
luisa_compute_add_test(test_dstorage                integration/runtime/test_dstorage.cpp)
luisa_compute_add_test(test_dstorage_decompression  integration/runtime/test_dstorage_decompression.cpp)
luisa_compute_add_test(test_procedural               integration/runtime/test_procedural.cpp)
luisa_compute_add_test(test_procedural_callable      integration/runtime/test_procedural_callable.cpp)
luisa_compute_add_test(test_rtx                      integration/runtime/test_rtx.cpp)
luisa_compute_add_test(test_oso_parser               integration/runtime/test_oso_parser.cpp)

if (LUISA_COMPUTE_ENABLE_GUI)
    luisa_compute_add_test(test_bindless               integration/runtime/test_bindless.cpp)
    luisa_compute_add_test(test_bindless_buffer         integration/runtime/test_bindless_buffer.cpp)
    luisa_compute_add_test(test_curve                   integration/runtime/test_curve.cpp)
    luisa_compute_add_test(test_curve_pbrt              integration/runtime/test_curve_pbrt.cpp)
    luisa_compute_add_test(test_curve_pbrt_diffuse      integration/runtime/test_curve_pbrt_diffuse.cpp)
    luisa_compute_add_test(test_denoiser                integration/runtime/test_denoiser.cpp)
    luisa_compute_add_test(test_indirect                integration/runtime/test_indirect.cpp)
    luisa_compute_add_test(test_indirect_rtx            integration/runtime/test_indirect_rtx.cpp)
    luisa_compute_add_test(test_motion_blur             integration/runtime/test_motion_blur.cpp)
    luisa_compute_add_test(test_native_include          integration/runtime/test_native_include.cpp)
    luisa_compute_add_test(test_present                 integration/runtime/test_present.cpp)
    luisa_compute_add_test(test_runtime                 integration/runtime/test_runtime.cpp)
    luisa_compute_add_test(test_select_device           integration/runtime/test_select_device.cpp)
    luisa_compute_add_test(test_texture3d               integration/runtime/test_texture3d.cpp)

    luisa_compute_add_executable(test_transient_resource
            integration/runtime/test_transient_resource.cpp
            integration/runtime/transient_resource_device/managed_first_fit.cpp
            integration/runtime/transient_resource_device/managed_first_fit.h
            integration/runtime/transient_resource_device/transient_resource_device.cpp
            integration/runtime/transient_resource_device/transient_resource_device.h)
    target_include_directories(test_transient_resource PRIVATE ./)
endif ()

# --- integration/runtime: DX-only tests ---
if (LUISA_COMPUTE_ENABLE_DX)
    luisa_compute_add_test(test_raster         integration/runtime/test_raster.cpp         ARGS dx)
    luisa_compute_add_test(test_memory_compact  integration/runtime/test_memory_compact.cpp ARGS dx)
    # test_work_graph disabled: work_graph headers not yet available
    # luisa_compute_add_test(test_work_graph      integration/runtime/test_work_graph.cpp     ARGS dx)
endif ()

# --- integration/runtime: HIP/HIPRT tests (AMD GPU only) ---
if (LUISA_COMPUTE_ENABLE_HIP)
    luisa_compute_add_test(test_hiprt_standalone integration/runtime/test_hiprt_standalone.cpp)
endif ()

# --- integration/ir: standalone tests ---
if (LUISA_COMPUTE_ENABLE_RUST)
    luisa_compute_add_test(test_autodiff        integration/ir/test_autodiff.cpp)
    luisa_compute_add_test(test_autodiff_full    integration/ir/test_autodiff_full.cpp)
    luisa_compute_add_test(test_ast2ir           integration/ir/test_ast2ir.cpp)
    luisa_compute_add_test(test_ast2ir_headless  integration/ir/test_ast2ir_headless.cpp)
    luisa_compute_add_test(test_ast2ir_ir2ast    integration/ir/test_ast2ir_ir2ast.cpp)

    if (LUISA_COMPUTE_ENABLE_GUI)
        luisa_compute_add_test(test_kernel_ir    integration/ir/test_kernel_ir.cpp)
    endif ()
endif ()
