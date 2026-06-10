#include <charconv>
#include <string_view>

#include "demo/Application.hpp"

namespace {
bool parseUint(std::string_view text, uint32_t& value) {
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, value);
    return ec == std::errc{} && ptr == end;
}

bool consumeValue(int& index, int argc, char* argv[], std::string_view option, uint32_t& outValue) {
    std::string_view arg = argv[index];
    if (arg.starts_with(option) && arg.size() > option.size() && arg[option.size()] == '=') {
        return parseUint(arg.substr(option.size() + 1), outValue);
    }
    if (arg == option && (index + 1) < argc) {
        ++index;
        return parseUint(argv[index], outValue);
    }
    return false;
}
} // namespace

int main(int argc, char* argv[]) {
    harmonia::App::Config config;
    config.title = "Hyperion — Real-Time Path Tracer";
    config.assetsDir = HYPERION_ASSETS_DIR;
    config.sceneFile = "cornell_classic.scene.toml";
    Application::DemoConfig demoConfig;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        uint32_t value = 0;
        if (consumeValue(i, argc, argv, "--spp", value)) {
            demoConfig.spp = value;
            demoConfig.sppExplicit = true;
        } else if (consumeValue(i, argc, argv, "--depth", value)) {
            demoConfig.maxDepth = value;
        } else if (harmonia::App::applyCommonArg(config, i, argc, argv)) {
            continue;
        }
    }

    Application app;
    return app.run(std::move(config), std::move(demoConfig));
}
