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

bool consumeString(int& index, int argc, char* argv[], std::string_view option, std::string& outValue) {
    std::string_view arg = argv[index];
    if (arg.starts_with(option) && arg.size() > option.size() && arg[option.size()] == '=') {
        outValue = std::string(arg.substr(option.size() + 1));
        return true;
    }
    if (arg == option && (index + 1) < argc) {
        outValue = argv[++index];
        return true;
    }
    return false;
}
} // namespace

int main(int argc, char* argv[]) {
    Application::Config config;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        uint32_t         value = 0;
        std::string      strValue;
        if (!arg.starts_with('-')) {
            // First non-flag argument is the scene file path.
            config.sceneFile = arg;
        } else if (consumeValue(i, argc, argv, "--spp", value)) {
            config.spp = value;
            config.sppExplicit = true;
        } else if (consumeValue(i, argc, argv, "--depth", value)) {
            config.maxDepth = value;
        } else if (consumeValue(i, argc, argv, "--width", value)) {
            config.width = value;
        } else if (consumeValue(i, argc, argv, "--height", value)) {
            config.height = value;
        } else if (consumeString(i, argc, argv, "--output", strValue)) {
            config.outputFile = strValue;
        } else if (arg == "--no-validation") {
            config.validation = false;
        }
    }

    auto app = Application::create(config);
    if (!app) {
        return app.error();
    }
    return (*app)->run();
}
