# PLAN — Hyperion

Hyperion is the offline path tracer and the unbiased reference renderer of a five-repository
family (slang-math → Aether → Harmonia → Hyperion and Theia). Theia is aligned to Hyperion: the
parity comparison measures a Theia image against a Hyperion reference image. This file lists the
known limitations, what follows next, the numbers a change must not regress, the release process
and the release history. How the renderer works and how it is used is in [README.md](README.md).

## Status

Hyperion is at v1.0.0 and pins Aether v1.0.0 and Harmonia v1.0.0. No implementation item
specific to Hyperion is open.

## Known limitations

| Limitation | Detail |
|------------|--------|
| Specular manifold sampling covers two cases | It is verified for a planar mirror and for closed smooth glass whose interior is a pure absorber. Ordinary path tracing handles everything else: scattering media, dispersion, rough specular surfaces, chains with more than one specular vertex, and receivers that follow a diffuse bounce. Curved casters are not excluded by the code but are not validated by the fixtures. Reservoir reuse does not exist in Hyperion. |
| Ellipsoid emitters | An emissive analytic sphere with a non-uniform instance scale is skipped by next-event estimation with a warning. |
| Cross-mode captures agree only up to noise | Two captures of the same animation time made by different routes (an output frame of `--sequence-frames`, and a single capture with `--frame`) use different sample indices, so their pixels differ at noise level. Repeating the same route reproduces the pixels exactly. |
| Distance haze is a sky-blend model | The in-scattered light of a `[haze]` scene is the environment map blurred to six degrees, the sun disc included. A sun disc with a physical radiance would make everything white in its direction where haze is present. The maps in use have their sun disc clipped, so the model is right for them (Harmonia's `PLAN.md` has the number). |

## What follows next

Nothing specific to Hyperion is planned. The active work of the family happens in the shared
library and in the real-time renderer; Hyperion takes it over through the dependency pins in
`CMakeLists.txt` and has to keep the properties of a reference renderer. The family's direction
is to adopt new light-transport techniques in the shared estimator and BSDF wherever both
renderers benefit, and in one renderer only where the renderers differ.

| Area | Where the work happens | What Hyperion has to keep |
|------|------------------------|---------------------------|
| BSDF, surface estimator, sampling | Harmonia | The result stays unbiased: no clamp, no special case, no loosened tolerance. The OpenPBR conformance tests in `tests/unit/test_bsdf.cpp` are updated together with a change to the shared BSDF. |
| Manifold sampling and reservoir reuse | Harmonia (shared solver and classifier), Theia (reuse) | `--sms-mode off` stays the plain path-traced reference. Hyperion holds only payload and scene adapters. |
| Parity comparison and thresholds | Harmonia | A reference image is rendered with 256 samples per pixel at 320×240. The budget changes only together with a newly measured set of thresholds. |
| Scene format, animation and deformation data | Aether | The scene loader contract (a camera placement is mandatory) and the shared scene hooks that `hyperion::Scene` implements. |
| Application host, capture and presentation | Harmonia | One owner of the frame history; a capture is the raw estimator result. |

Out of scope: multi-node and multi-GPU rendering. Hyperion runs on one local GPU, no such
infrastructure exists or is planned, and the idea is revisited only if a multi-GPU setup is
adopted.

## Numbers a change must not regress

| Measurement | Value | What it measures |
|-------------|-------|------------------|
| Test suites | 80 of 80 pass (current working tree) | Unit, module and integration suites of this repository |
| Parity comparison | 14 of 14 scenes pass: `mean_diff` ≤ 120, Rel-MSE ≤ 250, SSIM ≥ 0.6, luminance-histogram correlation ≥ 0.8 (320×240; Hyperion 256 samples per pixel, Theia 256 frames) | Agreement between Theia and the Hyperion reference under Harmonia's thresholds; these thresholds belong to this budget |
| Gallery | 30 of 30 PNGs byte-identical when regenerated at 1280×720 with 64 samples per pixel | Reproducibility of a capture and stability of the rendering path |
| Vulkan validation over the full gallery | 60 of 60 runs without a validation message or application warning | Valid Vulkan usage on every scene |
| Capture reproducibility | Same seed and settings: identical EXR pixels. Seeds 123 and 456: 13,556 of 19,200 Cornell pixels differ | The seed selects the sample stream; a capture is replayable |
| Progressive samples | 8 accumulated single samples differ from the first sample in 13,289 of 19,200 pixels | A capture does not repeat its first sample |
| Manifold-sampling references | Rendered with 4096 samples per pixel and split seeds; uncertainty at most 0.5% | Reference images for comparisons that involve manifold sampling |
| Sequence capture | Output frame 0 is bit-identical to a single capture of the same pose with the same seed; all frames of a sequence are bit-identical across runs | Pinning the animation time per output frame does not change the sampling |

## Sequence rendering

`--sequence-frames <N>` together with `--output` renders N animation frames to disk, one
accumulated image per animation frame, with the animation time pinned per output frame. Static
scenes are captured the same way; the frames are independent captures. The flags and the file
naming are described in [README.md, Sequence capture of an animated scene](README.md#sequence-capture-of-an-animated-scene).

Motion blur comes from Harmonia's `--shutter` and `--time-samples`: each sample of an output frame
is rendered at its own time inside the open shutter, and the output frame is the mean of its
samples. That makes a shuttered Hyperion frame the reference for the motion blur of Theia. On the
Aether scene `motion_blur`, a shuttered frame agrees with the mean of its instants rendered one
by one to 0.2 % of the luminance in the mover's trail.

## Release process

A change is done when the test suites pass, a representative headless render runs without a
validation message, and the build is warning-clean.

A release additionally needs:

- `verify-full`: build, test suites, `format-check`, clang-tidy and cppcheck.
- The parity comparison over Harmonia's scene manifest
  (`Harmonia/tools/render_and_validate.py` with `Harmonia/tools/validation_manifest.toml`).
- A Vulkan validation run over the full 30-scene gallery (`Harmonia/tools/check_vulkan_validation.py`).
- Regenerated screenshots for the README gallery at 1280×720 with 64 samples per pixel; they
  must be byte-identical to the tracked ones when nothing in the rendering path changed.

The release order follows the dependency direction: slang-math, Aether, Harmonia, Hyperion. The
Aether and Harmonia pins in `CMakeLists.txt` move in the release commit, and only in a
coordinated release of the family. Work is committed in the repository it belongs to; a push
happens only on request. Tags are pushed together with the commits, so
`git ls-remote --tags origin` equals `git tag -l`.

## Release history

The notes of the most recent releases are in [RELEASE_NOTES.md](RELEASE_NOTES.md). Releases
before v0.6.19 are recorded only in the git history.

| Version | Date | Content |
|---------|------|---------|
| Unreleased | — | Motion blur in sequence captures through Harmonia's `--shutter` and `--time-samples`: the accumulation is kept across the time samples of an output frame. Distance haze of a scene with a `[haze]` section: the ray-generation shader composites each sample through the shared `haze.slang` and the ray payload carries the distance to the primary surface. At 1024 samples per pixel on `haze`, the transmittance read back from five cubes matches the formula within 0.006. The lens glare of Harmonia (`--glare`) is in the PNG of a capture and leaves the EXR untouched, which is exact in Hyperion because a capture repeats exactly. An anisotropic material without a tangent map stretches its highlight along the texture coordinates (Harmonia's shared `surface.slang`); the gallery screenshots of `openpbr_specular` and `openpbr_metals` are rendered with the new frame |
| v1.0.0 | 2026-11-26 | Stable baseline, released with Harmonia v1.0.0 and Aether v1.0.0. Aether declared as an external include boundary in the architecture contract; static-analysis findings in the manifold-sampling and BSDF tests removed; documentation-reference check in strict mode |
| v0.12.0 | 2026-11-24 | Camera assembled by Harmonia's shared `CameraAssembly`; the rendered image is unchanged |
| v0.11.0 | 2026-11-23 | Documentation aligned with the code; static-analysis repairs |
| v0.10.0 | 2026-11-20 | Camera placement required: the private default view for Cornell scenes is deleted, and the shared scene loader rejects a scene without a camera |
| v0.9.0 | 2026-11-16 | Deformation (linear-blend and dual-quaternion skinning, morph targets) through Harmonia's shared `DeformPass`; camera bound to an animated node; frame seek and time-based sequence capture; one animation clock |
| v0.7.21 | 2026-11-07 | Pipeline binary cache applied to the ray-tracing pipeline: a pipeline without a usable driver key is built from its shaders; gallery re-rendered with the cache enabled, byte-identical |
| v0.7.20 | 2026-11-06 | Harmonia's Vulkan capability wave taken over through the dependency pin; no change to the renderer |
| v0.7.19 | 2026-11-05 | Specular manifold sampling with the shared compute sampler and the path classifier, and the reference mode without it (`--sms-mode off`) |
| v0.7.18 | 2026-11-03 | Sampling, seed and accumulation correctness: `--spp N` accumulates N single-sample captures, the seed reaches the shaders, the shared accumulation is the only history owner |
| v0.7.17 | 2026-10-30 | Dead-code check; one CMake target per module; gallery refreshed |
| v0.7.16 | 2026-10-27 | Dependency pins |
| v0.7.15 | 2026-10-17 | Lint configuration of the family (blocking lint job in CI); coverage script fixed |
| v0.7.14 | 2026-10-16 | `Scene` implements Harmonia's ray-tracing scene hooks and owns its TLAS; coverage lane; architecture contract replaces the earlier layering check |
| v0.7.13 | 2026-10-15 | Translation units open `namespace hyperion`; layering check |
| v0.7.12 | 2026-10-11 | Hyperion's classes move into `namespace hyperion`; gallery regenerated, byte-identical |
| v0.7.11 | 2026-10-09 | Continuous integration on Windows and Linux; descriptor ranges use the buffers' real sizes; gallery regenerated, pixel-identical |
| v0.7.10 | 2026-09-28 | Consistency tests between sampler and density for the bounded visible-normal sampling of the shared BSDF |
| v0.7.9 | 2026-09-26 | `--depth` honored; pipeline recursion depth corrected to the constant 2 |
| v0.7.8 | 2026-09-23 | OpenPBR 1.1.2 conformance of the shared BSDF (`specular_weight` above 1, transmission gray-shift) and new specification defaults, taken over through the dependency pins; conformance test added |
| v0.7.7 | 2026-08-02 | OpenPBR opacity cut-out with a shadow any-hit pair; four hit groups in the pipeline; opacity micromaps |
| v0.7.6 | 2026-08-02 | Harmonia's Vulkan capability adoption taken over; no change to Hyperion |
| v0.7.5 | 2026-08-01 | Next-event estimation for emissive analytic spheres; `--no-postfx`; unknown command-line arguments are errors; `VK_KHR_ray_tracing_maintenance1` becomes a hard requirement |
| v0.7.4 | 2026-07-30 | Denoiser switched off for `--output` captures |
| v0.7.3 | 2026-07-29 | Refactoring; 29-scene gallery regenerated; no change to the rendering |
| v0.7.2 | 2026-07-29 | Refactoring; no change to the behavior |
| v0.7.1 | 2026-07-26 | Fixes: emission added on the correct side of an emitter; reprojection |
| v0.7.0 | 2026-07-25 | v0.6.19 to v0.7.0: OpenPBR conformance of the shared BSDF checked against MaterialX |
