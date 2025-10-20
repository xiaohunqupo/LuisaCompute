target("lc-backend-dx")
set_basename("luisa-backend-dx")
_config_project({
    project_kind = "shared",
    batch_size = 8
})
add_deps("lc-runtime", "lc-vstl", "lc-hlsl-codegen")
add_files("**.cpp")
add_headerfiles("**.h", "../common/default_binary_io.h")
add_includedirs("./")
add_syslinks("dxgi")
if is_plat("windows") then
    add_defines("UNICODE", "_CRT_SECURE_NO_WARNINGS")
end
on_load(function(target)
    target:add("syslinks", "D3D12")
    target:add("defines", "LUISA_DX_SDK")
    if has_config("lc_enable_win_pix") then
        target:add("linkdirs", target:targetdir())
        target:add("links", "WinPixEventRuntime")
        target:add("defines", "LCDX_ENABLE_WINPIX")
    end
    if has_config("lc_enable_dxagsdk") then
        target:add("defines", "LCDX_ENABLE_AGILITY_SDK")
    end
    if has_config("lc_backend_lto") then
        target:set("policy", "build.optimization.lto", true)
        if is_config("lc_toolchain", "llvm") then
            target:add("ldflags", "-fuse-ld=lld-link")
            target:add("shflags", "-fuse-ld=lld-link")
        end
    end
    if has_config("lc_dx_cuda_interop") then
        import("detect.sdks.find_cuda")
        import("cuda_sdkdir", {
            rootdir = path.absolute(path.join(path.directory(os.scriptdir()), "cuda"))
        })
        local cuda = find_cuda(cuda_sdkdir())
        if not cuda then
            utils.error("cuda not found.")
        else
            local cuda_linkdirs = cuda["linkdirs"]
            target:add("linkdirs", cuda_linkdirs, {public = true})
            target:add("includedirs", cuda["includedirs"], {public = true})
            if is_plat("linux") and type(cuda_linkdirs) == "table" then
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
            target:add("defines", "LCDX_ENABLE_CUDA")
            target:add("syslinks", "Cfgmgr32", "Advapi32")
        end
    end
end)
set_pcxxheader("lc_dx_pch.h")
add_rules('lc_install_sdk', {
    libnames = 'dx_sdk'
})
target_end()
