local _sdks = {
    dx_sdk = {
        name = 'dx_sdk_20250816.zip',
    }
}

function sdk_address(sdk)
    return sdk['address'] or 'https://github.com/LuisaGroup/SDKs/releases/download/sdk/'
end
function sdk_mirror_addresses(sdk)
    return sdk['mirror_addresses'] or {}
end
function sdks()
    return _sdks
end
local lc_project_dir = path.directory(os.scriptdir())
function sdk_dir(arch, custom_dir)
    if custom_dir then
        if not path.is_absolute(custom_dir) then
            custom_dir = path.absolute(custom_dir, os.projectdir())
        end
    else
        custom_dir = path.join(lc_project_dir, 'SDKs/')
    end
    return path.join(custom_dir, arch)
end

function get_or_create_sdk_dir(arch, custom_dir)
    local dir = sdk_dir(arch, custom_dir)
    local lib = import('lib')
    lib.mkdirs(dir)
    return dir
end
