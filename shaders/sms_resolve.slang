import common;
import sms_work;
[[vk::push_constant]] SmsQueuePush pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint slot;
    if (!smsQueueValidationSlot(pc, group, lane, slot)) return;
    SmsWorkRecord first = pc.work[slot];
    uint samples = first.words[14].y;
    if (samples == 0u) { smsQueueCompletionError(pc, kSmsQueueInvalidWork); return; }
    if (slot % samples != 0u) return;
    uint width, height;
    hdrTarget.GetDimensions(width, height);
    uint pixel = slot / samples;
    if (width == 0u || pixel / width >= height || samples > pc.capacity-slot ||
        any(first.words[15].zw != uint2(width,height))) {
        smsQueueCompletionError(pc, kSmsQueueInvalidWork); return;
    }
    uint2 xy = uint2(pixel%width, pixel/width);
    float3 sum = hdrTarget[xy].xyz;
    for (uint s = 0u; s < samples; ++s) {
        SmsWorkRecord work = pc.work[slot+s];
        uint phase = smsPhase(work);
        if (phase != kSmsAbsent && phase != kSmsDone) {
            smsQueueCompletionError(pc, kSmsQueueInvalidWork); return;
        }
        if (phase == kSmsDone) sum += asfloat(work.words[11].xyz) * (asfloat(work.words[14].x)/float(samples));
    }
    if (!all(isfinite(sum))) { smsQueueCompletionError(pc, kSmsQueueNumericError); return; }
    // The following serial queue control pass observes errors and only THEN can
    // the waited host report Ready. Failed frames never enter accumulation.
    hdrTarget[xy] = float4(sum, 1.0);
}
