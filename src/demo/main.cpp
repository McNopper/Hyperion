#include <cstdint>

#include "demo/Application.hpp"
#include "harmonia/app/App.hpp"
#include "harmonia/core/Logger.hpp"

namespace {
bool consumeValue(int& index, int argc, char* const argv[], std::string_view option, std::uint32_t& outValue) {
    std::string_view arg = argv[index];
    if (arg.starts_with(option) && arg.size() > option.size() && arg[option.size()] == '=') {
        return harmonia::CliParser::parseUint32(arg.substr(option.size() + 1), outValue);
    }
    if (arg == option && (index + 1) < argc) {
        ++index;
        return harmonia::CliParser::parseUint32(argv[index], outValue);
    }
    return false;
}
} // namespace

int main(int argc, char* const argv[]) {
    harmonia::Logger::setTag("HYPERION");
    harmonia::App::Config config;
    config.title = "Hyperion — Path Tracer";
    config.width = 1024;
    config.height = 768;
    config.assetsDir = HYPERION_ASSETS_DIR;
    config.sceneFile = "cornell_classic.scene.toml";
    Application::DemoConfig demoConfig;

    for (int i = 1; i < argc; ++i) {
        std::uint32_t value = 0;
        if (consumeValue(i, argc, argv, "--spp", value)) {
            demoConfig.spp = value;
            demoConfig.sppExplicit = true;
        } else if (consumeValue(i, argc, argv, "--depth", value)) {
            demoConfig.maxDepth = value;
        } else if (harmonia::CliParser::applyCommonArg(config, i, argc, argv)) {
            continue;
        }
    }

    Application app;
    return app.run(std::move(config), std::move(demoConfig));
}
