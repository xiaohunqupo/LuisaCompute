local function test_proj(name)
    target(name)
    add_deps("lc-core", 'lc-vstl')
    _config_project({
        project_kind = "binary"
    })
    add_files(name .. ".cpp")
    add_includedirs("../common", {
        public = true
    })
    target_end()
end
test_proj("basic_traits")
test_proj("basic_types")
test_proj("binary_file_stream")
test_proj("binary_io")
test_proj("clock")
test_proj("dynamic_module")
test_proj("first_fit")
test_proj("logging")
test_proj("mathematics")
test_proj("pool")
test_proj("first_fit")
