includes("common")
if has_config("lc_dx_backend") then
    includes("dx")
end
if has_config("lc_cuda_backend") then
    includes("cuda")
end
if has_config("lc_metal_backend") then
    includes("metal")
end
if has_config("lc_fallback_backend") then
    includes("fallback")
end
if has_config("lc_vk_backend") then
    includes("vk")
end
includes("validation")
if has_config("lc_toy_c_backend") then
    includes("toy_c")
end

rule("lc-backend-deps")
on_load(function(target)
    target:add("deps", "lc-validation-layer", {
        inherit = false
    })
    -- target:add("deps", "lc-backend-toy-c", {
    --     inherit = false
    -- })
    if has_config("lc_dx_backend") then
        target:add("deps", "lc-backend-dx", "lc_install_dxsdk", {
            inherit = false
        })
    end
    if has_config("lc_cuda_backend") then
        target:add("deps", "lc-backend-cuda", {
            inherit = false
        })
    end
    if has_config("lc_metal_backend") then
        target:add("deps", "lc-backend-metal", {
            inherit = false
        })
    end
    if has_config("lc_vk_backend") then
        target:add("deps", "lc-backend-vk", {
            inherit = false
        })
    end
    if has_config("lc_fallback_backend") then
        target:add("deps", "lc-backend-fallback", {
            inherit = false
        })
    end
end)
rule_end()

target("lc-backends-dummy")
set_kind("phony")
add_rules("lc-backend-deps")
target_end()

function lc_make_dummy_backend(name, group_name)
    target("lc-backends-dummy-" .. name)
    set_kind("phony")
    set_group(group_name)
    add_rules("lc-backend-deps")
    target_end()
end
