# Hyperion v1.0.0-pre-alpha (2026-11-28) — prerelease

Pre-alpha development snapshot and GitHub backup. Consumes the shared OpenPBR/color-management
corrections in Harmonia v1.0.0-pre-alpha and applies the shared medium-exit radiance transport in
the ray-tracing adapter. Verification: **84/84** tests pass. The README gallery is regenerated for
this snapshot: 30/30 renders complete and all 30 tracked/staged/diff triplets are reviewed.
Selected native title official-pixel SSIM is **0.708761**; the optional agent-initiated native
top-down Hyperion experiment scores **0.436860**. Detailed OpenPBR Shader Playground validation
continues after the prerelease.

---

# Hyperion v1.0.0 (2026-11-26) — stable release

First stable release. It is released together with Harmonia v1.0.0 and Aether v1.0.0, the
versions it pins.

## Changes

- Aether is declared as an external include boundary in the architecture contract.
- Static-analysis findings in the manifold-sampling and BSDF tests are removed.
- The documentation-reference check runs in strict mode: every path that a document mentions
  must exist.

## Verification

- Test suites: 79 of 79 pass.
- `verify-full`, the architecture check, the dead-code check and the strict
  documentation-reference check report no finding.

---

# Hyperion v0.12.0 (2026-11-24)

Consumes Harmonia v0.12.0 and Aether v0.12.0.

## Changes

- The camera is assembled by Harmonia's shared `CameraAssembly`. The rendered image is unchanged.

## Verification

- Test suites: 79 of 79 pass. Harmonia's camera-assembly tests: 7 of 7 pass.
- A direct capture is identical before and after the change: `mean_diff` 0, `rel_mse` 0, SSIM 1.0,
  PSNR infinite.
- Parity comparison: 14 of 14 scenes pass. Vulkan validation over the full gallery: 60 of 60 runs
  without a message.

## Gallery

- README gallery regenerated at 1280×720 with 64 samples per pixel: 30 of 30 images byte-identical
  to the tracked gallery.

---

# Hyperion v0.9.0 (2026-11-16)

Consumes Harmonia v0.9.0 and Aether v0.9.0.

## Changes

- Deformation: skinned and morphed meshes are rendered through Harmonia's shared `DeformPass`
  (linear-blend and dual-quaternion skinning, morph targets). The estimator is unchanged.
- Parity thresholds recalibrated in Harmonia's scene manifest: `mean_diff` ≤ 120, Rel-MSE ≤ 250,
  SSIM ≥ 0.6, luminance-histogram correlation ≥ 0.8. A fresh run passes 14 of 14 scenes.
- Camera bound to an animated node (`[camera] node`): the camera is placed once per animation
  time. `--frame N` starts the animation at frame N; `--sequence-frames 1 --frame N` renders one
  accumulated image at animation frame N.
- One animation clock and time-based sequence capture: `--capture-every-ms` and `--start-seconds`
  together with `--sequence-frames`.

## Verification

- Unit suite: 43 of 43 pass.
- Sequence renders of the skin, morph and dual-quaternion fixtures: exit code 0 and no
  validation message; motion is visible (frame 0 against frame 2: `mean_diff` 10.1 to 10.3).
- `camera_orbit` rendered at `--frame 0` and `--frame 58` (Hyperion 256 samples per pixel, Theia
  64): the difference between the two frames is `mean_diff` 122.9 for Hyperion, within 1.7% of
  Theia's 124.9.
- An interval capture (`--capture-every-ms 33.33`) and a frame-grid capture at t = 1/30 s give the
  same image up to noise: mean difference 0.000, SSIM 1.000.
