target("lc-ast")
set_basename("luisa-ast")
_config_project({
	project_kind = "shared",
	batch_size = 4
})
add_deps("lc-core", "lc-vstl")
add_headerfiles("../../include/luisa/ast/**.h")
set_pcxxheader("lc_ast_pch.h")
add_files("**.cpp")
add_defines("LUISA_AST_EXPORT_DLL")
target_end()
