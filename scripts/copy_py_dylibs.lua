function main()
	if os.is_host("windows") then
		local dst = path.join(os.scriptdir(), "../src/py/luisa/dylibs")
		os.mkdir(dst)
		local async_opt = {
            copy_if_different = true,
            async = true,
            detach = true
        }
		os.cp(path.join(os.scriptdir(), "../bin/release/*.dll"), dst, async_opt)
		os.cp(path.join(os.scriptdir(), "../bin/release/*.pyd"), dst, async_opt)
	end
end
