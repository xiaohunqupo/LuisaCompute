target("lc-runtime")
set_basename("luisa-runtime")
_config_project({
	project_kind = "shared",
	batch_size = 8
})
add_deps("lc-ast")
set_pcxxheader("lc_runtime_pch.h")
add_defines("LUISA_RUNTIME_EXPORT_DLL")
if get_config("lc_enable_ir") then
	add_defines("LUISA_ENABLE_IR", {
		public = true
	})
	add_deps("lc-ir")
end
add_headerfiles("../../include/luisa/runtime/**.h")
add_files("**.cpp")
target_end()
