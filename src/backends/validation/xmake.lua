target("lc-validation-layer")
set_basename("luisa-validation-layer")
_config_project({
	project_kind = "shared"
})
set_pcxxheader("lc_validation_pch.h")
add_deps("lc-runtime", "lc-vstl")
add_files("**.cpp")
add_headerfiles("**.h")
target_end()
