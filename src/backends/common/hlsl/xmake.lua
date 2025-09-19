target("lc-hlsl-codegen")
_config_project({
    project_kind = "static",
    batch_size = 2
})
add_deps("lc-vstl")
add_files("*.cpp")
set_pcxxheader("lc_hlsl_pch.h")
add_headerfiles("*.h")
on_load(function(target)
    if get_config("lc_no_hlsl_builtin") then
        target:add("defines", "LC_NO_HLSL_BUILTIN")
    else
        target:add("files", path.join(os.scriptdir(), "builtin/*.cpp"))
    end
end)
target_end()
