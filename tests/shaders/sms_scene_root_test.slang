import common;
import sms_transport;
import sms_scene;
struct RootInput { float4 receiver, light, seed; };
struct RootOutput { float4 first, second, metrics; };
struct RootPush { RootInput* inputs; RootOutput* outputs; uint count, pad; };
[[vk::push_constant]] RootPush pc;

[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pc.count) return;
    RootInput input = pc.inputs[id.x];
    HyperionSmsScene scene;
    SmsManifoldVertex reference[2], root[2];
    reference[0] = reference[1] = (SmsManifoldVertex)0;
    reference[0].event = reference[1].event = kSmsTransmission;
    bool solved = scene.project(input.receiver.xyz,input.seed.xyz,reference,2u,root);
    if (solved) solved = solveSmsManifold(scene,input.receiver.xyz,input.light.xyz,root,2u,32u,1.0e-5);
    float geometric = 0.0, residual = 0.0;
    if (solved) solved = smsGeometricTerm(input.receiver.xyz,root,2u,input.light.xyz,
        float3(1,0,0),float3(0,1,0),geometric) &&
        smsManifoldResidual(input.receiver.xyz,root,2u,input.light.xyz,residual);
    RootOutput output = {};
    output.first = float4(root[0].position,solved ? 1.0 : 0.0);
    output.second = float4(root[1].position,solved ? 1.0 : 0.0);
    output.metrics = float4(geometric,residual,root[0].etaTOverI,root[1].etaTOverI);
    pc.outputs[id.x] = output;
}
