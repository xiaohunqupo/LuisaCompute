target("lc-tensor")
_config_project({
	project_kind = "shared"
})
set_pcxxheader("lc_tensor_pch.h")
add_deps("lc-ast", "lc-runtime", "lc-vstl", "lc-dsl")
add_headerfiles("../../include/luisa/tensor/**.h")
add_files("**.cpp")
-- add_defines("LUISA_TENSOR_STATIC_LIB", {
--     public = true
-- })

add_defines("LUISA_TENSOR_EXPORT_DLL")