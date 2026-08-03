// texsampler.slang — Hyperion's bindless texture fetch for the shared surface module.
//
// Lives in its own module because more than one hit stage needs it: the
// closest-hit populates the full OpenPBR surface frame, and the shadow any-hit
// samples `map_opacity` alone. Both must read the texture the same way (mip 0,
// combined image sampler) or the cutout would differ between camera and shadow
// rays.

import common;
import surface;

/// Bindless-texture fetcher for the shared surface-frame populator (mip 0).
struct HyperionTextureSampler : ISurfaceTextureSampler {
    float4 sampleTex(uint idx, float2 uv) {
        return bindlessTextures[NonUniformResourceIndex(idx)]
            .SampleLevel(bindlessSamplers[NonUniformResourceIndex(idx)], uv, 0.0);
    }
};
