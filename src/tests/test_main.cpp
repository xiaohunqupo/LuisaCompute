#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "config.h"

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/logging.h>

namespace luisa::test {

static luisa::vector<const char *> args;

int argc() noexcept { return static_cast<int>(args.size()); }
const char *const *argv() noexcept { return args.data(); }

static luisa::vector<const char *> _backends;
int backends_to_test_count() noexcept {
    return static_cast<int>(std::min<size_t>(_backends.size(), static_cast<size_t>(std::numeric_limits<int>::max())));
}
const char *const *backends_to_test() noexcept {
    return _backends.data();
}

inline void args_filter(int argc_in, const char **argv_in) noexcept {
    args.clear();
    _backends.clear();
    bool default_backend = true;
    for (int i = 0; i < argc_in; ++i) {
        if (!luisa::string_view{argv_in[i]}.starts_with("--backend-")) {
            args.emplace_back(argv_in[i]);
        } else {
            default_backend = false;
            _backends.emplace_back(argv_in[i] + 10);
        }
    }
    if (default_backend) {
        _backends.emplace_back("dx");
        _backends.emplace_back("cuda");
    }
    args.emplace_back(nullptr);
}

}// namespace luisa::test

int main(int argc, const char **argv) {
    luisa::test::args_filter(argc, argv);
    return 0;
}
