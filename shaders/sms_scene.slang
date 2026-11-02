// Production scene reader for SMS. Unlike RT builtins, all reconstruction runs
// in the common compute trial context; no initial root is generated in closesthit.
import common;
import math;
import sms_transport;

struct HyperionSmsHit {
    SmsManifoldVertex vertex;
    uint primitive;
    float distance;
};

float3 smsUnitDerivative(float3 n, float lengthBeforeNormalization, float3 derivative) {
    return (derivative - n * dot(n, derivative)) / lengthBeforeNormalization;
}

struct HyperionSmsScene : ISmsProjector {
    bool trace(float3 origin, float3 direction, float maximum, out HyperionSmsHit hit) {
        hit = {};
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = kMediumEpsilon;
        ray.TMax = maximum;
        // Nearest alpha geometry blocks this supported opaque path class. It is
        // never skipped and then counted again by ordinary PT's null interactions.
        RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
        q.TraceRayInline(tlas, RAY_FLAG_NONE, kInstanceMaskAll, ray);
        while (q.Proceed()) {
            if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
                GpuInstance instance = instances[q.CandidateInstanceID()];
                if (instance.geometryKind != 1u) continue;
                float3 o = q.CandidateObjectRayOrigin(), d = q.CandidateObjectRayDirection();
                float a = dot(d, d), b = dot(o, d);
                float c = dot(o, o) - instance.sphereRadius * instance.sphereRadius;
                float discriminant = b * b - a * c;
                if (!(a > 0.0 && discriminant >= 0.0)) continue;
                float root = sqrt(discriminant);
                float t = (-b - root) / a;
                if (t <= ray.TMin) t = (-b + root) / a;
                if (t > ray.TMin && t < maximum) q.CommitProceduralPrimitiveHit(t);
            }
        }
        uint status = q.CommittedStatus();
        if (status != COMMITTED_TRIANGLE_HIT && status != COMMITTED_PROCEDURAL_PRIMITIVE_HIT) return false;
        uint instanceId = q.CommittedInstanceID();
        GpuInstance instance = instances[instanceId];
        float3x4 o2w = q.CommittedObjectToWorld3x4();
        float3x4 w2o = q.CommittedWorldToObject3x4();
        float3x3 normalMatrix = transpose((float3x3)w2o);
        hit.distance = q.CommittedRayT();
        hit.primitive = q.CommittedPrimitiveIndex();
        SmsManifoldVertex v = {};
        v.position = origin + direction * hit.distance;
        v.objectKey = uint2(instanceId, instance.materialIndex);
        v.etaTOverI = 1.0;
        if (status == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
            float3 objectPoint = mul(w2o, float4(v.position, 1.0));
            float3 raw = mul(normalMatrix, objectPoint);
            float magnitude = length(raw);
            if (!(magnitude > 0.0)) return false;
            v.normal = v.geoNormal = raw / magnitude;
            float3x3 frame = buildFrameFromZ(v.geoNormal);
            v.dpdu = frame[0]; v.dpdv = frame[1];
            v.dndu = smsUnitDerivative(v.normal, magnitude,
                mul(normalMatrix, mul((float3x3)w2o, v.dpdu)));
            v.dndv = smsUnitDerivative(v.normal, magnitude,
                mul(normalMatrix, mul((float3x3)w2o, v.dpdv)));
        } else {
            uint first = instance.indexOffset + hit.primitive * 3u;
            GpuVertex a = vertices[indices[first]], b = vertices[indices[first + 1u]],
                c = vertices[indices[first + 2u]];
            float3 p0 = mul(o2w, float4(a.position, 1.0));
            float3 e1 = mul(o2w, float4(b.position, 1.0)) - p0;
            float3 e2 = mul(o2w, float4(c.position, 1.0)) - p0;
            float3 crossEdges = cross(e1, e2);
            float area2 = length(crossEdges);
            if (!(area2 > 0.0)) return false;
            v.geoNormal = crossEdges / area2;
            v.dpdu = normalize(e1);
            v.dpdv = cross(v.geoNormal, v.dpdu);
            float3 n0 = mul(normalMatrix, a.normal), n1 = mul(normalMatrix, b.normal),
                n2 = mul(normalMatrix, c.normal);
            n0 = dot(n0,n0) > 0.0 ? normalize(n0) : v.geoNormal;
            n1 = dot(n1,n1) > 0.0 ? normalize(n1) : v.geoNormal;
            n2 = dot(n2,n2) > 0.0 ? normalize(n2) : v.geoNormal;
            float2 bary = q.CommittedTriangleBarycentrics();
            float3 raw = n0 * (1.0 - bary.x - bary.y) + n1 * bary.x + n2 * bary.y;
            float magnitude = length(raw);
            if (!(magnitude > 0.0)) return false;
            float sign = dot(raw, v.geoNormal) < 0.0 ? -1.0 : 1.0;
            raw *= sign;
            v.normal = raw / magnitude;
            // Barycentric gradients on the WORLD triangle's tangent plane.
            float d11 = dot(e1,e1), d22 = dot(e2,e2), d12 = dot(e1,e2);
            float determinant = dot(crossEdges,crossEdges);
            float3 g1 = (d22 * e1 - d12 * e2) / determinant;
            float3 g2 = (d11 * e2 - d12 * e1) / determinant;
            float3 dn1 = (n1 - n0) * sign, dn2 = (n2 - n0) * sign;
            v.dndu = smsUnitDerivative(v.normal, magnitude,
                dn1 * dot(g1,v.dpdu) + dn2 * dot(g2,v.dpdu));
            v.dndv = smsUnitDerivative(v.normal, magnitude,
                dn1 * dot(g1,v.dpdv) + dn2 * dot(g2,v.dpdv));
        }
        hit.vertex = v;
        return all(isfinite(v.position)) && all(isfinite(v.normal)) &&
            all(isfinite(v.dndu)) && all(isfinite(v.dndv));
    }

    bool supportedEvent(float3 previous, inout SmsManifoldVertex v, uint event) {
        GpuMaterial mat = materials[v.objectKey.y];
        bool front = dot(previous - v.position, v.geoNormal) > 0.0;
        if (event == kSmsReflection) {
            if (!smsReflectionCaster(mat) || !front) return false;
            v.etaTOverI = 1.0;
        } else {
            if (!smsSmoothGlass(mat)) return false;
            float eta = smsGlassEta(mat);
            if (!(eta > 1.0)) return false;
            v.etaTOverI = front ? eta : 1.0 / eta;
            if (!front && smsPureAbsorption(mat)) {
                // The existing medium boundary handler uses raw GEOMETRIC
                // normals, unlike the ordinary surface BSDF's shading frame.
                v.normal = v.geoNormal;
                if (instances[v.objectKey.x].geometryKind == 0u)
                    v.dndu = v.dndv = 0.0.xxx; // triangle geometric normals are piecewise constant
                // Analytic spheres already carry geometric-normal derivatives.
            }
        }
        v.event = event;
        return true;
    }

    [mutating]
    bool project(float3 receiver, float3 proposedFirst, SmsManifoldVertex reference[2],
        uint count, out SmsManifoldVertex projected[2]) {
        projected[0] = projected[1] = (SmsManifoldVertex)0;
        float3 edge = proposedFirst - receiver;
        if (!(dot(edge, edge) > 0.0)) return false;
        HyperionSmsHit first;
        if (!trace(receiver, normalize(edge), kInf, first)) return false;
        projected[0] = first.vertex;
        if (!supportedEvent(receiver, projected[0], reference[0].event)) return false;
        if (count == 2u) {
            if (projected[0].event != kSmsTransmission || projected[0].etaTOverI <= 1.0) return false;
            float3 wo = normalize(receiver - projected[0].position);
            float3 n = projected[0].normal;
            if (dot(n, wo) < 0.0) n = -n;
            float3 wi; float weight;
            if (!smsDeltaInterface(wo, n, kSmsTransmission, 1.0 / projected[0].etaTOverI, wi, weight)) return false;
            HyperionSmsHit second;
            if (!trace(projected[0].position, wi, kInf, second)) return false;
            projected[1] = second.vertex;
            if (!supportedEvent(projected[0].position, projected[1], kSmsTransmission) ||
                any(projected[0].objectKey != projected[1].objectKey) || projected[1].etaTOverI >= 1.0) return false;
        }
        return true;
    }
};
