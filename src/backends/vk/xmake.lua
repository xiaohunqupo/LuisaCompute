target("lc-backend-vk")
_config_project({
    project_kind = "shared"
})
add_deps("lc-runtime", "lc-vstl", "lc-hlsl-codegen")
add_headerfiles("*.h", "../common/default_binary_io.h")
add_files("*.cpp")
set_pcxxheader("lc_vk_pch.h")
-- TODO: use dxc for vulkan, only windows temporarily
if is_plat("windows") then
    add_defines("VK_USE_PLATFORM_WIN32_KHR")
elseif is_plat("linux") then
    add_defines("VK_USE_PLATFORM_XCB_KHR")
end
add_defines("USE_SPIRV")
on_load(function(target)
    target:add("deps", "volk")
    if get_config("lc_no_hlsl_builtin") then
        target:add("defines", "LC_NO_HLSL_BUILTIN")
    end
end)
target_end()
