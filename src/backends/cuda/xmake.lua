if has_config("lc_cuda_ext_lcub") then
    includes("lcub")
end
target("lc-cuda-base")
set_kind("phony")
on_load(function(target)
    import("detect.sdks.find_cuda")
    import("cuda_sdkdir")
    local cuda = find_cuda(cuda_sdkdir())
    if cuda then
        local cuda_linkdirs = cuda["linkdirs"]
        target:add("linkdirs", cuda_linkdirs, {
            public = true
        })
        if target:is_plat("linux") and type(cuda_linkdirs) == "table" then
            for _, v in ipairs(cuda_linkdirs) do
                local stubs_dir = path.join(v, "stubs")
                if os.exists(stubs_dir) then
                    target:add("linkdirs", stubs_dir, {
                        public = true
                    })
                end
            end
        end
        target:add("includedirs", cuda["includedirs"], {
            public = true
        })
    else
        target:set("enabled", false)
        return
    end
    if target:is_plat("windows") then
        target:add("defines", "UNICODE", "_CRT_SECURE_NO_WARNINGS", {
            public = true
        })
        target:add("syslinks", "Cfgmgr32", "Advapi32", {
            public = true
        })
    end
    if has_config("lc_backend_lto") then
        target:set("policy", "build.optimization.lto", true)
        if is_config("lc_toolchain", "llvm") then
            target:add("ldflags", "-fuse-ld=lld-link")
            target:add("shflags", "-fuse-ld=lld-link")
        end
    end
end)
target_end()

target("lc-backend-cuda")
set_basename("luisa-backend-cuda")
_config_project({
    project_kind = "shared",
    batch_size = 4
})
add_deps("lc-runtime", "lc-cuda-base")

add_deps("lc_embed_codegen", {
    inherit = false,
    public = false
})
add_rules("lc_compile_codegen", {
    remove_ext = true,
    remove_slash_r = true,
    var_name_prefix = "luisa_compute_"
})
add_files("cuda_builtin.lua")
set_pcxxheader("lc_cuda_pch.h")
add_headerfiles("*.h")
on_load(function(target)
    if has_config("lc_reproc_use_xrepo") then
        target:add("packages", "reproc")
    else
        target:add("deps", "reproc")
    end
    if has_config("lc_cuda_ext_lcub") then
        target:add("deps", "lc-compute-cuda-ext-lcub")
    end
    target:add("headerfiles", path.normalize(path.join(os.scriptdir(), "../common/default_binary_io.h")))
    local src_path = os.scriptdir()
    local exclude_files = {}
    exclude_files["cuda_nvrtc_compiler.cpp"] = true
    exclude_files["cuda_builtin_embedded.cpp"] = true
    exclude_files["cuda_devrt_embedded.cpp"] = true
    for _, filepath in ipairs(os.files(path.join(src_path, "*.cpp"))) do
        local file_name = path.filename(filepath)
        if not exclude_files[file_name] then
            target:add("files", filepath)
        end
    end
    target:add("defines", "LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN")
    target:add("deps", "lc-vulkan-swapchain", "lc-volk")
end)
add_files("extensions/cuda_denoiser.cpp", "extensions/cuda_dstorage.cpp", "extensions/cuda_pinned_memory.cpp")
add_links("cuda")

-- after_build(function(target)
--     import("lib.detect.find_file")
--     import("detect.sdks.find_cuda")
--     import("cuda_sdkdir")
--     local cuda = find_cuda(cuda_sdkdir())
--     if cuda then
--         local linkdirs = cuda["linkdirs"]
--         local lc_bin_dir = target:targetdir()
--         if is_plat("windows") then
--             for i, v in ipairs(linkdirs) do
--                 os.cp(path.join(v, "cudadevrt.lib"), lc_bin_dir)
--             end
--         end
--         -- TODO: linux
--     end
-- end)
target_end()

target("lc-nvrtc")
_config_project({
    project_kind = "binary",
    runtime = "MT"
})
if is_plat("windows") then
    add_syslinks("Ws2_32", "User32")
end
set_basename("luisa_nvrtc")
add_deps("lc-cuda-base")
add_links("nvrtc_static", "nvrtc-builtins_static", "nvptxcompiler_static")
add_files("cuda_nvrtc_compiler.cpp")
target_end()
