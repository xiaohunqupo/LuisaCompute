local function test_proj(name, callable)
    local amd_name = "amd_" .. name
    lc_make_dummy_backend(amd_name, amd_name)
    target(amd_name)
    set_group(amd_name)
    _config_project({
        project_kind = "binary"
    })
    add_files(name .. ".cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "stb-image")
    add_deps("lc-gui")
    if callable then
        callable()
    end
    target_end()
end

test_proj("test_copy")
test_proj("test_compress_dstorage")
