function main()
	if os.is_host("windows") then
		local dst = path.join(os.scriptdir(), "../src/py/luisa/dylibs")
		os.mkdir(dst)
		os.cp(path.join(os.scriptdir(), "../bin/release/*.dll"), dst, {copy_if_different = true})
		os.cp(path.join(os.scriptdir(), "../bin/release/*.pyd"), dst, {copy_if_different = true})
	end
end
