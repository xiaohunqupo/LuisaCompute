target("lc-api")
set_basename("luisa-api")
_config_project({
	project_kind = "shared"
})
add_deps("lc-runtime")
add_files("**.cpp")
add_includedirs("../rust")
target_end()
