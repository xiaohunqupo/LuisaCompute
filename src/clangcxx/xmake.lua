if not is_mode("debug") then
    target("lc-clangcxx")
    set_basename("luisa-clangcxx")
    _config_project({
        project_kind = "shared"
    })
    set_pcxxheader("src/lc_clangcxx_pch.h")
    add_files("src/**.cpp")
    on_load(function(target, opt)
        target:add("headerfiles", path.normalize(path.join(os.scriptdir(), "../common/default_binary_io.h")))
        local libs = {}
        local lc_llvm_path = get_config("lc_llvm_path")
        if (not lc_llvm_path) or (lc_llvm_path == "") then
            lc_llvm_path = path.join(os.scriptdir(), "llvm")
        end
        local p = path.join(lc_llvm_path, "lib/*.lib")
        target:add("linkdirs", path.join(lc_llvm_path, "lib"))
        target:add("includedirs", path.join(lc_llvm_path, "include"))
        local black_list = {
            "lld",
        }
        for __, filepath in ipairs(os.files(p)) do
            local basename = path.basename(filepath)
            for _, v in ipairs(black_list) do
                if basename:match(v) ~= nil then
                    goto END_LOOP
                end
            end
            table.insert(libs, basename)
            ::END_LOOP::
        end
        target:add("links", libs)
        target:add("defines", "LUISA_CLANGCXX_EXPORT_DLL", 'CLANG_BUILD_STATIC')
        target:add("deps", "lc-core", "lc-runtime", "lc-vstl")
        if is_plat("windows") then
            target:add("syslinks", "Version", "advapi32", "Shcore", "user32", "shell32", "Ole32", 'Ws2_32', 'ntdll', {
                public = true
            })
        elseif is_plat("linux") then
            target:add("syslinks", "uuid")
        elseif is_plat("macosx") then
            target:add("frameworks", "CoreFoundation")
        end
        if is_mode("release") then
            target:add("defines", "LC_CLANGCXX_ENABLE_COMMENT=0")
        else
            target:add("defines", "LC_CLANGCXX_ENABLE_COMMENT=1")
        end

    end)
    target_end()
end
