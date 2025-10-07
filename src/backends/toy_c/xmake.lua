target("luisa-backend-toy-c")
_config_project({
	project_kind = "shared"
})
add_deps("luisa-runtime", "luisa-vstl", "luisa-clanguage-codegen")
add_files("*.cpp")
add_headerfiles("**.h")
set_pcxxheader("lc_toy_c_pch.h")
target_end()
