import common;
import env;

[shader("miss")]
void backgroundMiss(inout RayPayload payload)
{
    // Write G-buffer sky sentinel for primary rays.
    if (payload.depth == 0u) {
        const uint2 pix = DispatchRaysIndex().xy;
        gNormal[pix] = float4(0.0, 0.0, 0.0, 0.0);
        gDepth[pix]  = kInf;
    }
    payload.radiance += payload.throughput * evalEnvironment(WorldRayDirection(), pushConst.envLuminanceScale);
    payload.terminated = true;
}
