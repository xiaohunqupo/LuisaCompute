if not has_config("lc_glfw_use_xrepo") then
    target("glfw")
    set_basename("luisa-ext-glfw")
    _config_project({
        project_kind = "static"
    })

    on_load(function(target)
        local function rela(p)
            return path.normalize(path.join(os.scriptdir(), p))
        end
        target:add("headerfiles", rela("../ext/glfw/include/**.h"))
        target:add("files", rela("../ext/glfw/src/*.c"))
        target:add("includedirs", rela("../ext/glfw/include"), {
            public = true
        })

        if target:is_plat("linux") then
            target:add("defines", "_GLFW_X11", "_DEFAULT_SOURCE")
        elseif target:is_plat("windows") then
            target:add("defines", "_GLFW_WIN32")
            target:add("syslinks", "User32", "Gdi32", "Shell32")
        elseif target:is_plat("macosx") then
            target:add("files", path.normalize(path.join(os.scriptdir(), "../ext/glfw/src/*.m")))
            target:add("mflags", "-fno-objc-arc")
            target:add("defines", "_GLFW_COCOA")
            target:add("frameworks", "Foundation", "Cocoa", "IOKit", "OpenGL", "QuartzCore")
        end
    end)
    target_end()
end
if has_config("lc_enable_imgui") and not has_config("lc_imgui_use_xrepo") then
    target("imgui")
    set_basename("luisa-ext-imgui")
    _config_project({
        project_kind = "shared"
    })
    on_load(function(target)
        local function rela(p)
            return path.normalize(path.join(os.scriptdir(), p))
        end
        target:add("headerfiles", rela("../ext/imgui/*.h"), rela("../ext/imgui/backends/*.h"))
        target:add("files", rela("../ext/imgui/*.cpp"), rela("../ext/imgui/backends/imgui_impl_glfw.cpp"))
        target:add("includedirs", rela("../ext/imgui"), rela("../ext/imgui/backends"), {
            public = true
        })
        if target:is_plat("windows") then
            target:add("defines", "IMGUI_API=__declspec(dllexport)");
            target:add("defines", "IMGUI_API=__declspec(dllimport)", {
                interface = true
            });
        elseif target:is_plat("linux") then
            target:add("syslinks", "X11")
        end
        if has_config("lc_glfw_use_xrepo") then
            target:add("packages", "glfw")
        else
            target:add("deps", "glfw")
        end
    end)

    add_defines("ImDrawIdx=unsigned int", "GLFW_INCLUDE_NONE", "IMGUI_DEFINE_MATH_OPERATORS", {
        public = true
    })
    -- add_deps("lc-dsl")
    target_end()
end

target("lc-gui")
set_basename("luisa-gui")
_config_project({
    project_kind = "shared"
})
on_load(function(target)
    local function rela(p)
        return path.normalize(path.join(os.scriptdir(), p))
    end
    target:add("headerfiles", rela("../../include/luisa/gui/**.h"))
    if has_config("lc_glfw_use_xrepo") then
        target:add("packages", "glfw")
    else
        target:add("deps", "glfw")
    end
    if has_config("lc_enable_imgui") then
        if has_config("lc_imgui_use_xrepo") then
            target:add("packages", "imgui")
        else
            target:add("deps", "imgui")
        end
        target:add("files", path.join(os.scriptdir(), "*.cpp"))
    else
        for _, filepath in ipairs(os.files(path.join(os.scriptdir(), "*.cpp"))) do
            if filepath:match("imgui") == nil then
                target:add("files", filepath)
            end
        end
        target:add("files", path.join(os.scriptdir(), "*.cpp|imgui_window.cpp"))
    end
end)
add_defines("LUISA_GUI_EXPORT_DLL")
add_deps("lc-runtime", "lc-dsl")
target_end()
