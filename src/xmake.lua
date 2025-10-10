enable_mimalloc = has_config("lc_enable_mimalloc")
enable_custom_malloc = has_config("lc_enable_custom_malloc")
table.insert(_config_rules, "lc-rename-ext")
local rename_rule_idx = table.getn(_config_rules)
includes("ext/EASTL", "ext/spdlog", "ext/reproc", "ext/liblmdb", "ext/volk", "ext/stb")
-- yyjson
do
    target("lc-yyjson")
    _config_project({
        project_kind = "static"
    })
    on_load(function(target)
        local src_path = path.join(os.scriptdir(), "ext/yyjson/src")
        target:add("files", path.join(src_path, "yyjson.c"))
        target:add("includedirs", src_path, {
            public = true
        })
    end)
    target_end()
end
table.remove(_config_rules, rename_rule_idx)
includes("core", "vstl", "runtime")
if has_config("lc_enable_osl") then
    includes("osl")
end
if has_config("lc_enable_dsl") then
    includes("dsl")
end
if has_config("lc_enable_gui") then
    includes("gui")
end
if has_config("_lc_enable_py") then
    includes("py")
end
includes("backends")
if has_config("lc_enable_tests") then
    includes("tests")
end
if has_config("lc_enable_clangcxx") then
    includes("clangcxx")
end

-- includes("tensor")
