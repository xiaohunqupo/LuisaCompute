includes("common")
if get_config("lc_dx_backend") then
    includes("dx")
end
if get_config("lc_cuda_backend") then
    includes("cuda")
end
if get_config("lc_metal_backend") then
    includes("metal")
end
if get_config("lc_cpu_backend") then
    includes("cpu")
end
if LCRemoteBackend then
    includes("remote")
end
if get_config("lc_vk_backend") then
    includes("vk")
end
includes("validation")
if get_config("lc_toy_c_backend") then
    includes("toy_c")    
end
target("lc-backends-dummy")
set_kind("phony")
on_load(function(target)
    target:add("deps", "luisa-validation-layer", {
        inherit = false
    })
    -- target:add("deps", "luisa-backend-toy-c", {
    --     inherit = false
    -- })
    if get_config("lc_dx_backend") then
        target:add("deps", "luisa-backend-dx", {
            inherit = false
        })
    end
    if get_config("lc_cuda_backend") then
        target:add("deps", "luisa-backend-cuda", {
            inherit = false
        })
    end
    if get_config("lc_metal_backend") then
        target:add("deps", "luisa-backend-metal", {
            inherit = false
        })
    end
    if get_config("lc_vk_backend") then
        target:add("deps", "luisa-backend-vk", {
            inherit = false
        })
    end
    if get_config("lc_cpu_backend") then
        target:add("deps", "luisa-backend-cpu", {
            inherit = false
        })
    end
end)
target_end()
