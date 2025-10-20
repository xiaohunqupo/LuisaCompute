local rootdir = {
    rootdir = path.join(os.scriptdir(), 'scripts')
}
local packages = import("packages", rootdir)
local find_sdk = import("find_sdk", rootdir)
local lib = import("lib", rootdir)
function main(custom_dir)
    if os.is_host("windows") then
        find_sdk.install_sdk('dx_sdk', custom_dir)
    end
end
