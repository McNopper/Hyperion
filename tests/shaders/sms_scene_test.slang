import common;
import sms_scene;
struct ReaderInput { float4 origin, direction; };
struct ReaderOutput { float4 position, normal, geometric, dpdu, dpdv, dndu, dndv, keys; };
struct ReaderPush { ReaderInput* inputs; ReaderOutput* outputs; uint count, pad; };
[[vk::push_constant]] ReaderPush pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pc.count) return;
    HyperionSmsScene scene;
    HyperionSmsHit hit;
    float maximum=pc.inputs[id.x].direction.w>0.0 ? pc.inputs[id.x].direction.w : kInf;
    ReaderOutput result = {};
    if (pc.pad==1u) {
        result.position.w=scene.occluded(pc.inputs[id.x].origin.xyz,normalize(pc.inputs[id.x].direction.xyz),maximum) ? 1.0 : 0.0;
        pc.outputs[id.x]=result; return;
    }
    bool found = scene.trace(pc.inputs[id.x].origin.xyz, normalize(pc.inputs[id.x].direction.xyz), kInf, hit);
    if (found) {
        result.position = float4(hit.vertex.position, 1.0);
        result.normal = float4(hit.vertex.normal, 0.0);
        result.geometric = float4(hit.vertex.geoNormal, 0.0);
        result.dpdu = float4(hit.vertex.dpdu, 0.0);
        result.dpdv = float4(hit.vertex.dpdv, 0.0);
        result.dndu = float4(hit.vertex.dndu, 0.0);
        result.dndv = float4(hit.vertex.dndv, 0.0);
        result.keys = float4(float(hit.vertex.objectKey.x), float(hit.vertex.objectKey.y), float(hit.primitive), hit.distance);
    }
    pc.outputs[id.x] = result;
}
