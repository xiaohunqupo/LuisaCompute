target("lc-tensor")
set_basename("luisa-tensor")
_config_project({
	project_kind = "shared"
})
lc_set_pcxxheader("lc_tensor_pch.h")
add_deps("lc-runtime", "lc-vstl", "lc-dsl")
add_headerfiles("../../include/luisa/tensor/**.h")
add_files("**.cpp")
-- add_defines("LUISA_TENSOR_STATIC_LIB", {
--     public = true
-- })

add_defines("LUISA_TENSOR_EXPORT_DLL")