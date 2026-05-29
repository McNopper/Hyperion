import common;
import math;

/// Sample the equirectangular IBL env map for the given direction (Rec.2020 linear).
float3 sampleEnvMap(float3 dir)
{
    float3 d = normalize(dir);
    float u = atan2(d.z, d.x) * (0.5 * kInvPi) + 0.5;
    float v = acos(clamp(d.y, -1.0, 1.0)) * kInvPi;  // v=0 at top (y=+1), v=1 at bottom (y=-1)
    return max(envMap.SampleLevel(envSampler, float2(u, v), 0).xyz, 0.0.xxx);
}

/// Evaluate environment radiance for a given direction (Rec.2020 linear, cd/m²).
/// Returns envLuminanceScale * IBL sample when an env map is loaded,
/// or black (0,0,0) for scenes without an env map.
float3 evalEnvironment(float3 direction, float envLuminanceScale)
{
    if (pushConst.hasEnvMap != 0u) {
        return envLuminanceScale * sampleEnvMap(direction);
    }
    return 0.0.xxx;
}
