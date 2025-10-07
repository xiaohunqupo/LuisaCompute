target("luisa-tensor")
_config_project({
	project_kind = "shared"
})
set_pcxxheader("lc_tensor_pch.h")
add_deps("luisa-ast", "luisa-runtime", "luisa-vstl", "luisa-dsl")
add_headerfiles("../../include/luisa/tensor/**.h")
add_files("**.cpp")
-- add_defines("LUISA_TENSOR_STATIC_LIB", {
--     public = true
-- })

add_defines("LUISA_TENSOR_EXPORT_DLL")