local function test_proj(name, extra_deps)
    target(name)
    add_deps("lc-core", 'lc-vstl')
    if extra_deps then
        for _, dep in ipairs(extra_deps) do
            add_deps(dep)
        end
    end
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
test_proj("test_mimalloc")