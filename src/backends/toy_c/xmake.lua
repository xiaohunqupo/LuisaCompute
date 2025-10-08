target("lc-backend-toy-c")
set_basename("luisa-backend-toy-c")
_config_project({
	project_kind = "shared"
})
add_deps("lc-runtime", "lc-vstl", "lc-clanguage-codegen")
add_files("*.cpp")
add_headerfiles("**.h")
set_pcxxheader("lc_toy_c_pch.h")
target_end()
