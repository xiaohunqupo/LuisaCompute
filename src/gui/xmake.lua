target("glfw")
set_basename("luisa-ext-glfw")
_config_project({
    project_kind = "static"
})
add_headerfiles("../ext/glfw/include/**.h")
add_files("../ext/glfw/src/*.c")
add_includedirs("../ext/glfw/include", {
    public = true
})
on_load(function(target)
    target:add("defines", "_GLFW_BUILD_DLL")
    if target:is_plat("linux") then
        target:add("defines", "_GLFW_X11", "_DEFAULT_SOURCE")
    elseif target:is_plat("windows") then
        target:add("defines", "_GLFW_WIN32")
        target:add("syslinks", "User32", "Gdi32", "Shell32")
    elseif target:is_plat("macosx") then
        target:add("files", path.translate(path.join(os.scriptdir(), "../ext/glfw/src/*.m")))
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
    if target:is_plat("windows") then
        target:add("defines", "IMGUI_API=__declspec(dllexport)");
        target:add("defines", "IMGUI_API=__declspec(dllimport)", {
            interface = true
        });
    elseif target:is_plat("linux") then
        target:add("syslinks", "X11")
    end
end)
add_headerfiles("../ext/imgui/*.h", "../ext/imgui/backends/*.h")
add_files("../ext/imgui/*.cpp", "../ext/imgui/backends/imgui_impl_glfw.cpp")
add_includedirs("../ext/imgui", "../ext/imgui/backends", {
    public = true
})
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
add_headerfiles("../../include/luisa/gui/**.h")
add_files("*.cpp")
add_defines("LUISA_GUI_EXPORT_DLL", "GLFW_DLL")
add_deps("lc-runtime", "imgui")
target_end()
