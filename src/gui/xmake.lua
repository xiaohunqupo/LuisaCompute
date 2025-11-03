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

    target:add("defines", "_GLFW_BUILD_DLL")
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
end)

add_defines("ImDrawIdx=unsigned int", "GLFW_INCLUDE_NONE", "IMGUI_DEFINE_MATH_OPERATORS", {
    public = true
})
add_deps("glfw", "lc-dsl")
target_end()

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
end)
add_files("*.cpp")
add_defines("LUISA_GUI_EXPORT_DLL", "GLFW_DLL")
add_deps("lc-runtime", "imgui")
target_end()
