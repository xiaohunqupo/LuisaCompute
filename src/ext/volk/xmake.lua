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
    target:add("includedirs", path.join(os.scriptdir(), "include"), {
        public = true
    })
    if target:is_plat("linux") then
        target:add("defines", "VK_USE_PLATFORM_XLIB_KHR", {
            public = true
        })
    end
end)
target_end()
