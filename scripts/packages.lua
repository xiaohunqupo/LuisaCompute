local _sdks = {
    dx_sdk = {
        -- from:
        --> xmake l hash.sha256 SDKs/x64/dx_sdk_20240920.zip
        sha256 = "2b0c2cd4bbee2544c1f5130a327287119ce183b2e9cdf4bb0d70b8ccc7975c16",
        name = 'dx_sdk_20250614.zip',
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
function sdk_dir(arch, custom_dir)
    if custom_dir then
        if not path.is_absolute(custom_dir) then
            custom_dir = path.absolute(custom_dir, os.projectdir())
        end
    else
        custom_dir = path.join(os.projectdir(), 'SDKs/')
    end
    return path.join(custom_dir, arch)
end

function get_or_create_sdk_dir(arch, custom_dir)
    local dir = sdk_dir(arch, custom_dir)
    local lib = import('lib')
    lib.mkdirs(dir)
    return dir
end
