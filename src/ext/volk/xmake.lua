if get_config("_lc_vk_sdk_dir") then
    target("volk")
    _config_project({
        project_kind = "static"
    })
    add_includedirs(".", {
        public = true
    })
    add_defines("VK_NO_PROTOTYPES", {
        public = true
    })
    add_files("volk.c")
    on_load(function(target)
        if is_plat("windows") then
            local lc_sdk_dir = get_config("_lc_vk_sdk_dir")
            target:add("includedirs", path.join(lc_sdk_dir, "Include"), {
                public = true
            })
        end
        if is_plat("linux") then
            target:add("defines", "VK_USE_PLATFORM_XLIB_KHR", {public = true})
        end
    end)
    target_end()
end
