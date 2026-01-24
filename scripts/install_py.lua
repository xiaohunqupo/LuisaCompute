function main()
    local dylib_src_path = path.normalize(path.join(os.scriptdir(), '../bin/release'))
    local py_src_path = path.normalize(path.join(os.scriptdir(), '../src/py/luisa'))
    local dylib_dst_path = path.normalize(path.join(os.scriptdir(), '../bin/py/luisa/dylibs'))
    os.mkdir(dylib_dst_path)
    local copy_option = {
        async = true,
        copy_if_different = true,
        detach = true
    }
    os.cp(path.join(dylib_src_path, '**.dll'), dylib_dst_path, copy_option)
    os.cp(path.join(dylib_src_path, '**.so'), dylib_dst_path, copy_option)
    os.cp(path.join(dylib_src_path, 'lcapi.pyd'), dylib_dst_path, copy_option)
    os.cp(path.join(py_src_path, '*.py'), path.directory(dylib_dst_path), copy_option)
    print("Copy test cases? (y/n)")
    local option = io.read()
    if option == 'y' or option == 'Y' then
        local test_path = path.normalize(path.join(os.scriptdir(), '../src/tests/python/*.py'))
        local dst_path = path.normalize(path.join(dylib_dst_path, '../..'))
        os.cp(test_path, dst_path, copy_option)
    end
end
