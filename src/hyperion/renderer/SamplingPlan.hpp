#ifndef HYPERION_RENDERER_SAMPLINGPLAN_HPP
#define HYPERION_RENDERER_SAMPLINGPLAN_HPP

#include <algorithm>
#include <cstdint>

namespace hyperion {

struct SamplingPlan {
    std::uint32_t frameCount;
    std::uint32_t samplesPerDispatch;
};

/// Capture --spp N means N total samples, not N frames of N samples. Interactive
/// batching still traces N samples per logical frame; the host alone owns history.
[[nodiscard]] constexpr SamplingPlan samplingPlan(std::uint32_t requested, bool capture) noexcept {
    const auto count = std::max(requested, 1U);
    return capture ? SamplingPlan{count, 1U} : SamplingPlan{1U, count};
}

} // namespace hyperion
#endif // HYPERION_RENDERER_SAMPLINGPLAN_HPP
