import("net.http")
import("utils.archive")
import("lib.detect.find_file")
import("core.project.config")
local _sdks = {
    dx_sdk = {
        name = 'dx_sdk_20250816.zip'
    }
}

function sdk_address(sdk)
    local address = sdk['address']
    if address == nil then
        return 'https://github.com/LuisaGroup/SDKs/releases/download/sdk/'
    end
    if #address == 0 then -- no download
        return nil
    end
    return address
end
function sdk_mirror_addresses(sdk)
    return sdk['mirror_addresses'] or {}
end
function sdks()
    return _sdks
end
local lc_project_dir = path.directory(os.scriptdir())
function sdk_dir(custom_dir)
    if custom_dir then
        if not path.is_absolute(custom_dir) then
            custom_dir = path.absolute(custom_dir, os.projectdir())
        end
    else
        custom_dir = path.join(lc_project_dir, 'SDKs/')
    end
    return path.join(custom_dir, os.host(), os.arch())
end

function get_or_create_sdk_dir(custom_dir)
    local dir = sdk_dir(custom_dir)
    local lib = import('lib')
    lib.mkdirs(dir)
    return dir
end

-- use_lib_cache = true

local function try_download(zip, url, mirror_urls, dst_dir, settings)
    local function check_file_valid()
        local zip_dir = find_file(zip, {dst_dir})
        return zip_dir ~= nil
    end
    http.download(url .. zip, dst_dir, {
        continue = false
    })
    if check_file_valid() then
        return
    end
    for i, mirror_url in ipairs(mirror_urls) do
        http.download(mirror_url .. zip, dst_dir, {
            continue = false
        })
        if check_file_valid() then
            return
        end
    end
    utils.error("Download " .. zip .. " failed, please check your internet.")
end

function file_from_github(sdk_map, dir)
    local zip, address, mirror_address
    zip = sdk_map['name']
    address = sdk_address(sdk_map)
    if not address then
        return
    end
    mirror_address = sdk_mirror_addresses(sdk_map)

    local zip_dir = find_file(zip, {dir})
    local dst_dir = path.join(dir, zip)
    if (zip_dir == nil) then
        local url = vformat(address)
        print("downloading: " .. url .. zip .. " to: " .. dir)
        try_download(zip, url, mirror_address, dst_dir, {
            continue = false
        })
    end
end

-- tool

function find_tool_zip(tool_name, dir)
    local zip_dir = find_file(tool_name, {dir})
    return {
        name = tool_name,
        dir = zip_dir
    }
end

function unzip_sdk(tool_name, in_dir, out_dir)
    local zip_file = find_tool_zip(tool_name, in_dir)
    if (zip_file.dir ~= nil) then
        print("install: " .. zip_file.name)
        archive.extract(zip_file.dir, out_dir)
    else
        utils.error("failed to install " .. tool_name .. ", file " .. zip_file.name .. " not found!")
    end
end

function install_sdk(sdk_map, custom_dir)
    local dir = get_or_create_sdk_dir(custom_dir)
    local _sdks = sdks()
    if not sdk_map then
        utils.error("Invalid sdk: " .. sdk_map["name"])
        return
    end
    file_from_github(sdk_map, dir)
end

function check_file(sdk_name, custom_dir)
    local dir = sdk_dir(custom_dir)
    local _sdks = sdks()
    local sdk_map = _sdks[sdk_name]
    local zip = sdk_map['name']
    local zip_dir = find_file(zip, {dir})
    return zip_dir ~= nil
end
