import common;

[shader("intersection")]
void sphereIntersection()
{
    uint instanceIndex = (uint)InstanceID();
    GpuInstance instance = instances[instanceIndex];
    float3 center = vertices[instance.vertexOffset].position;
    float radius = instance.sphereRadius;

    float3 ro = WorldRayOrigin() - center;
    float3 rd = WorldRayDirection();
    float tMin = RayTMin();
    float tMax = RayTCurrent();

    float a = dot(rd, rd);
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float disc = b * b - a * c;
    if (disc < 0.0) {
        return;
    }

    float s = sqrt(disc);
    float invA = 1.0 / max(a, 1.0e-8);
    float t0 = (-b - s) * invA;
    float t1 = (-b + s) * invA;
    float t = t0;
    if (t < tMin || t > tMax) {
        t = t1;
    }
    if (t < tMin || t > tMax) {
        return;
    }

    SphereIntersectionAttributes attr;
    attr.t = t;
    ReportHit(t, 0u, attr);
}
