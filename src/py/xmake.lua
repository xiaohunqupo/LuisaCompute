target("lcapi")
on_load(function(target)
	local function split_str(str, chr, func)
		for part in string.gmatch(str, "([^" .. chr .. "]+)") do
			func(part)
		end
	end
	local lc_py_include = get_config("lc_py_include")
	split_str(lc_py_include, ';', function(v)
		target:add("includedirs", v)
	end)
	local lc_py_linkdir = get_config("lc_py_linkdir")
	local lc_py_libs = get_config("lc_py_libs")
	if type(lc_py_linkdir) == "string" then
		split_str(lc_py_linkdir, ';', function(v)
			target:add("linkdirs", v)
		end)
	end
	if type(lc_py_libs) == "string" then
		split_str(lc_py_libs, ';', function(v)
			target:add("links", v)
		end)
	end
	local function rela(p)
		return path.relative(path.absolute(p, os.scriptdir()), os.projectdir())
	end
	target:add("includedirs", rela("../ext/stb/"), rela("../ext/pybind11/include"))
end)

_config_project({
	project_kind = "shared",
	enable_exception = true
})
set_pcxxheader("lcpy_pch.h")
add_cxflags("/bigobj", {
	tools = "cl"
})
add_headerfiles("*.h")
add_files("*.cpp")
add_deps("luisa-runtime", "luisa-gui")
set_extension(".pyd")
target_end()
