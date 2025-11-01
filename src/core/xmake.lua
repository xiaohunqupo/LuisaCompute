target("lc-core")
set_basename("luisa-core")
_config_project({
    project_kind = "shared",
    batch_size = 4
})
on_load(function(target)
    local lib = import("lib", {rootdir = get_config("_lc_script_path")})
    local function rela(p)
        return lib.lexically_normal(path.join(os.scriptdir(), p))
    end
    target:add("includedirs", rela("../../include"), rela("../ext/xxHash/"), rela("../ext/magic_enum/include"),
        rela("../ext/half/include"), {
            public = true
        })
    if target:is_plat("windows") then
        if is_mode("debug") then
            target:add("syslinks", "Dbghelp")
        end
        target:add("defines", "NOMINMAX", "LUISA_PLATFORM_WINDOWS", "_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR", {
            public = true
        })
    elseif target:is_plat("linux") then
        target:add("defines", "LUISA_PLATFORM_UNIX", {
            public = true
        })
    elseif target:is_plat("macosx") then
        target:add("defines", "LUISA_PLATFORM_UNIX", "LUISA_PLATFORM_APPLE", {
            public = true
        })
    end
    if has_config("lc_enable_dsl") then
        target:add("defines", "LUISA_ENABLE_DSL", {
            public = true
        })
    end
    if has_config("lc_use_system_stl") then
        target:add("defines", "LUISA_USE_SYSTEM_STL", "_ENABLE_EXTENDED_ALIGNED_STORAGE", {public = true})
    end
    target:add("defines", "LUISA_CORE_EXPORT_DLL")
    if target:is_plat("windows") then
        target:add("defines", "_CRT_SECURE_NO_WARNINGS")
    end
    target:add("deps", "eastl", "spdlog", "lc-check-winsdk")
    if has_config("spdlog_only_fmt") then -- Use no spdlog
        target:add("defines", "LUISA_CUSTOM_LOGGER", {
            public = true
        })
    end
    local marl_path = path.join(os.scriptdir(), "../ext/marl")
    if (not has_config("lc_external_marl")) and (os.exists(marl_path)) then
        target:add("defines", "MARL_DLL", {
            public = true
        })
        target:add("defines", "MARL_BUILDING_DLL")
        target:add("files", path.join(marl_path, "src/*.c"), path.join(marl_path, "src/build.marl.cpp"))
        if not target:is_plat("windows") then
            target:add("files", path.join(marl_path, "src/*.S"))
        end
        target:add("includedirs", path.join(marl_path, "include"), {
            public = true
        })
    end
end)
add_headerfiles("../../include/luisa/core/**.h", "../ext/xxHash/**.h", "../ext/magic_enum/include/**.hpp",
    "../ext/half/include/half.hpp") -- , "../ext/parallel-hashmap/**.h"
add_files("**.cpp")
target_end()
