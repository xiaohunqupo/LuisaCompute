target("luisa-osl")
_config_project({
	project_kind = "shared",
	batch_size = 16
})
add_defines("LUISA_OSL_EXPORT_DLL")
add_deps("luisa-ast", "luisa-runtime")
add_headerfiles("../../include/luisa/osl/**.h")
add_files("**.cpp")
