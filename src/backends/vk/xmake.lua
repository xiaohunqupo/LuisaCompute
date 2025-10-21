target("lc-backend-vk")
set_basename("luisa-backend-vk")
_config_project({
    project_kind = "shared",
    batch_size = 8
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
on_load(function(target)
    target:add("deps", "volk")
    if has_config("lc_vk_cuda_interop") then
        import("detect.sdks.find_cuda")
        import("cuda_sdkdir", {
            rootdir = path.join(path.directory(os.scriptdir()), "cuda")
        })
        local cuda = find_cuda(cuda_sdkdir())
        if not cuda then
            utils.error("cuda not found.")
        else
            local cuda_linkdirs = cuda["linkdirs"]
            target:add("linkdirs", cuda_linkdirs, {public = true})
            target:add("includedirs", cuda["includedirs"], {public = true})
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

            target:add("links", "nvrtc_static", "cudart_static", "cuda")
            target:add("defines", "LUISA_VULKAN_ENABLE_CUDA_INTEROP")
        end
    end
end)
target_end()
