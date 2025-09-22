local script_dir = path.join(os.scriptdir(), "builtin")
local dst_script_dir = path.join(os.scriptdir(), "builtin_embed")
if not os.exists(dst_script_dir) then
    os.mkdir(dst_script_dir)
end
function src_dir()
    if get_config("lc_no_hlsl_builtin") then
        return nil
    end
    return script_dir
end

function dst_dir()
    return dst_script_dir
end

function meta_dir()
    return path.join(dst_script_dir, "_meta.msi")
end

function file_lists()
    return {'hlsl_header', 'dx_linalg', 'hlsl_header_fallback', 'raytracing_header', 'tex2d_bindless', 'tex3d_bindless',
            'compute_quad', 'determinant', 'inverse', 'indirect', 'resource_size', 'accel_header', 'copy_sign',
            'bindless_common', 'auto_diff', "reduce", 'accel_process_vk.dxil', 'load_bdls.dxil', 'load_bdls_vk.dxil',
            'set_accel4.dxil', 'bc6_encodeblock.dxil', 'bc6_trymodeg10.dxil', 'bc6_trymodele10.dxil',
            'bc7_encodeblock.dxil', 'bc7_trymode02.dxil', 'bc7_trymode137.dxil', 'bc7_trymode456.dxil'}
end
