target("lc-dsl")
set_basename("luisa-dsl")
_config_project({
	project_kind = "static",
	batch_size = 0
})
lc_set_pcxxheader("lc_dsl_pch.h")
add_deps("lc-runtime")
add_headerfiles("../../include/luisa/dsl/**.h")
add_files("**.cpp")
add_defines("LUISA_DSL_STATIC_LIB", {
    public = true
})