target("luisa-validation-layer")
_config_project({
	project_kind = "shared"
})
set_pcxxheader("lc_validation_pch.h")
add_deps("luisa-runtime", "luisa-vstl")
add_files("**.cpp")
add_headerfiles("**.h")
target_end()
