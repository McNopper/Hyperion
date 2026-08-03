import common;

/// The shadow ray reached `tMax` without a fully-present surface accepting it.
/// Whatever cut-out crossings it passed already multiplied their (1-α) into
/// `transmittance` in the any-hit; reaching here is what makes that product
/// meaningful (an accepted hit ends traversal instead, and the caller reads 0).
[shader("miss")]
void shadowMiss(inout ShadowPayload payload)
{
    payload.missed = true;
}
