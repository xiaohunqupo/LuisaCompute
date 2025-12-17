#include <luisa/version.h>
#include <luisa/core/dll_export.h>

LUISA_EXPORT_API int backend_version() { return LUISA_COMPUTE_VERSION; }
