import common;

[shader("miss")]
void shadowMiss(inout ShadowPayload payload)
{
    payload.inShadow = false;
}
