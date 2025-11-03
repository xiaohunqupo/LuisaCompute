target("lc-vstl")
set_basename("luisa-vstl")
local config_tb = {
    batch_size = 4
}
if is_plat("macosx") then
    config_tb["project_kind"] = "shared"
else
    config_tb["project_kind"] = "static"
    add_defines("LUISA_VSTL_STATIC_LIB", {
        public = true
    })
end
_config_project(config_tb)
add_deps("lc-core")
set_pcxxheader("lc_vstl_pch.h")
add_headerfiles("../../include/luisa/vstl/**.h")
add_files("**.cpp")
on_load(function(target)
    if has_config("lc_use_xrepo") then
        target:add("packages", "lmdb")
    else
        target:add("deps", "lmdb")
    end

    if target:is_plat("windows") then
        target:add("syslinks", "Ole32", {
            public = true
        })
    elseif target:is_plat("linux") then
        target:add("syslinks", "uuid", {
            public = true
        })
    elseif target:is_plat("macosx") then
        target:add("frameworks", "CoreFoundation", {
            public = true
        })
    end
end)
target_end()
