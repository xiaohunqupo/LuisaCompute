-- use command:
-- xmake lua printer_text_arr.lua
-- to execute this script and gen new files
-- 'accel_process', 'accel_process_vk', 'bindless_upload', 'bindless_upload_vk', 'bc6_encode_block', 'bc6_header', 'bc6_trymode_g10cs','bc6_trymode_le10cs', 'bc7_encode_block', 'bc7_header', 'bc7_trymode_02cs', 'bc7_trymode_137cs', 'bc7_trymode_456cs'
local files_list = {'cuda_builtin_kernels', 'cuda_device_half', 'cuda_device_math', 'cuda_device_resource'}
local file_ext = {".cu"}
local lib = import("lib")

local cuda_builtin_path = path.join(os.projectdir(), "src/backends/cuda/cuda_builtin")
local dst_path = path.join(os.projectdir(), "src/backends/cuda")
local write_file_name = "cuda_builtin_embedded"

function main()
    local test_zip_dir = path.join(os.projectdir(), "bin/release/test_zip.exe")
    local ss = lib.StringBuilder()
    local impl_sb = lib.StringBuilder()
    local header_sb = lib.StringBuilder()
    for i, file in ipairs(files_list) do
        -- make this file ignored by git
        local compressed_file = file .. ".msi"
        local curr_ext = ".h"
        if i <= #file_ext then
            curr_ext = file_ext[i] 
        end
        local file_with_ext = file .. curr_ext
        os.runv(test_zip_dir, {path.join(cuda_builtin_path, file_with_ext), path.join(cuda_builtin_path, compressed_file), "y"})
        local uncompressed_size
        try {function()
            local ff = io.open(path.join(cuda_builtin_path, file_with_ext), "rb")
            uncompressed_size = ff:size()
        end}
        local f = io.open(path.join(cuda_builtin_path, compressed_file), "rb")
        ss:clear()
        ss:add(f:read("*a"))
        f:close()
        impl_sb:add('extern "C" const unsigned char luisa_cuda_builtin_'):add(file):add("["):add(tostring(math.tointeger(
            ss:size()))):add("]={")
        tostring(math.tointeger(lib.to_byte_array(ss, impl_sb)))
        impl_sb:add("};\n")
        header_sb:add('extern "C" const unsigned char luisa_cuda_builtin_'):add(file):add("["):add(tostring(math.tointeger(
            ss:size()))):add("];\n"):add('constexpr uint64_t luisa_cuda_builtin_'):add(file):add("_size = "):add(tostring(math.tointeger(uncompressed_size))):add("ull;\n")

    end
    ss:dispose()
    impl_sb:write_to(path.join(dst_path, write_file_name .. ".cpp"))
    header_sb:write_to(path.join(dst_path, write_file_name .. ".h"))
    impl_sb:dispose()
    header_sb:dispose()
end
