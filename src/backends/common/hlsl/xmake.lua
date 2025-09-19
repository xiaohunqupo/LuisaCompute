target("lc-hlsl-codegen")
_config_project({
    project_kind = "static",
    batch_size = 2
})
add_deps("lc-vstl")
add_files("**.cpp")
set_pcxxheader("lc_hlsl_pch.h")
add_headerfiles("*.h")
target_end()