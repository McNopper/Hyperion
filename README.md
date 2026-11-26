# Hyperion

[![Build](https://github.com/McNopper/Hyperion/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/McNopper/Hyperion/actions/workflows/build.yml)

Vulkan path tracer for OpenPBR.

> *[Hyperion](https://en.wikipedia.org/wiki/Hyperion_(mythology)) — Titan of heavenly light, father of Helios, Selene and Eos.*

Hyperion is a GPU path tracer built on Vulkan 1.4 KHR ray tracing. It implements the
[OpenPBR Surface v1.1.2](https://academysoftwarefoundation.github.io/OpenPBR/) material model and
renders a physically correct, linearly encoded HDR image. It is the reference renderer of the
repository family it belongs to: the images it produces are the ground truth that the real-time
renderer Theia is measured against.

This page is the entry point to the documentation.

| Document | Content |
|----------|---------|
| [README.md](README.md) | This page: what Hyperion is, how it works, how to build, run, test and change it |
| [ARCHITECTURE.md](ARCHITECTURE.md) | The architecture contract that `tools/check_arch.py` checks the code against |
| [PLAN.md](PLAN.md) | Known limitations, what comes next, the release process, the release history |
| [RELEASE_NOTES.md](RELEASE_NOTES.md) | Notes of the most recent releases |
| [tools/README.md](tools/README.md) | The scripts in `tools/`: purpose, invocation, output |
| [AGENTS.md](AGENTS.md) | Short orientation for coding tools that read a repository description automatically |

---

## Screenshots

All 30 gallery scenes, rendered at 64 samples per pixel and 1280×720. The sample count is kept
low so that the whole gallery can be rendered in one pass; the remaining fireflies are the
expected noise at 64 samples per pixel.

| cornell_classic | cornell_classic_rec709 | cornell_empty |
|:---:|:---:|:---:|
| ![cornell_classic](screenshots/cornell_classic.png) | ![cornell_classic_rec709](screenshots/cornell_classic_rec709.png) | ![cornell_empty](screenshots/cornell_empty.png) |

| cornell_spheres | cornell_suzanne | cornell_textured_cube |
|:---:|:---:|:---:|
| ![cornell_spheres](screenshots/cornell_spheres.png) | ![cornell_suzanne](screenshots/cornell_suzanne.png) | ![cornell_textured_cube](screenshots/cornell_textured_cube.png) |

| openpbr_metals | openpbr_dielectrics | openpbr_coat |
|:---:|:---:|:---:|
| ![openpbr_metals](screenshots/openpbr_metals.png) | ![openpbr_dielectrics](screenshots/openpbr_dielectrics.png) | ![openpbr_coat](screenshots/openpbr_coat.png) |

| openpbr_fuzz | openpbr_specular | openpbr_organics |
|:---:|:---:|:---:|
| ![openpbr_fuzz](screenshots/openpbr_fuzz.png) | ![openpbr_specular](screenshots/openpbr_specular.png) | ![openpbr_organics](screenshots/openpbr_organics.png) |

| openpbr_thinfilm | openpbr_advanced | shaderball_base |
|:---:|:---:|:---:|
| ![openpbr_thinfilm](screenshots/openpbr_thinfilm.png) | ![openpbr_advanced](screenshots/openpbr_advanced.png) | ![shaderball_base](screenshots/shaderball_base.png) |

| shaderball_coat | shaderball_fuzz | shaderball_metal |
|:---:|:---:|:---:|
| ![shaderball_coat](screenshots/shaderball_coat.png) | ![shaderball_fuzz](screenshots/shaderball_fuzz.png) | ![shaderball_metal](screenshots/shaderball_metal.png) |

| shaderball_opacity | shaderball_specular | shaderball_subsurface |
|:---:|:---:|:---:|
| ![shaderball_opacity](screenshots/shaderball_opacity.png) | ![shaderball_specular](screenshots/shaderball_specular.png) | ![shaderball_subsurface](screenshots/shaderball_subsurface.png) |

| shaderball_thinfilm | shaderball_transmission | shaderball_emission |
|:---:|:---:|:---:|
| ![shaderball_thinfilm](screenshots/shaderball_thinfilm.png) | ![shaderball_transmission](screenshots/shaderball_transmission.png) | ![shaderball_emission](screenshots/shaderball_emission.png) |

| shaderball_checker | shader_ball | bunny_shaderball |
|:---:|:---:|:---:|
| ![shaderball_checker](screenshots/shaderball_checker.png) | ![shader_ball](screenshots/shader_ball.png) | ![bunny_shaderball](screenshots/bunny_shaderball.png) |

| camera_suzanne | ABeautifulGame | dragon_teapot |
|:---:|:---:|:---:|
| ![camera_suzanne](screenshots/camera_suzanne.png) | ![ABeautifulGame](screenshots/ABeautifulGame.png) | ![dragon_teapot](screenshots/dragon_teapot.png) |

---

## Architecture

Hyperion is one of five repositories that form a rendering pipeline. Dependencies point one way
only:

```mermaid
flowchart LR
    SM["slang-math<br/>math"] --> A["Aether<br/>file format"]
    SM --> H
    A --> H["Harmonia<br/>shared Vulkan lib"]
    H --> Hy["<b>Hyperion</b><br/>path tracer · ground truth"]
    H --> T["Theia<br/>real-time renderer"]
```

| Repository | Role |
|------------|------|
| [slang-math](https://github.com/McNopper/slang-math) | Math library: vector, quaternion and matrix types |
| [Aether](https://github.com/McNopper/Aether) | File formats: scenes, materials, cameras and presets (TOML), asset loaders |
| [Harmonia](https://github.com/McNopper/Harmonia) | Shared Vulkan library: application host, scene loading, shaders, estimators, tools |
| **Hyperion** | This repository: the offline path tracer, the reference renderer |
| [Theia](https://github.com/McNopper/Theia) | Real-time accumulation path tracer (rasterizer and ray tracing) |

### Why Hyperion is the reference renderer

A path tracer estimates the color of each pixel by simulating light transport with random
samples: for every pixel it follows light paths through the scene and averages the radiance they
collect. The estimate is noisy at low sample counts and converges to the physically correct image
as the sample count grows. Hyperion is an offline renderer: it takes the time a clean image
needs, and it uses none of the approximations a real-time renderer needs to stay interactive:

- The estimator has no clamp on bright values. A firefly clamp would bias bright contributions;
  the only guards set a negative or non-finite sample to zero, so that one broken path cannot
  contaminate the accumulation.
- Russian roulette (the random early termination of dark paths) is always active. It shortens
  paths without changing the expected result.
- A captured image never passes through the denoiser. The denoiser runs in the interactive
  window only.
- A capture is reproducible: the same `--rng-seed` and settings produce the same pixels.

The output is therefore an unbiased estimate of the true image. Theia is aligned to it: a
correction to the shared shading code lands in Hyperion first, the reference images are
regenerated, and Theia is then updated to match.

### Host and renderer

Hyperion consumes Aether and Harmonia through CMake `FetchContent`, pinned to release tags in
`CMakeLists.txt`. The executable is a thin subclass of Harmonia's application host
`harmonia::app::App`. The host owns the window, swapchain, HDR target, accumulation, tonemapping,
presentation and scene loading; Hyperion plugs its renderer in through the
`harmonia::app::IRenderer` interface.

Hyperion provides its own GPU scene layout for ray tracing: index buffers, one bottom-level
acceleration structure (BLAS) per mesh and one top-level acceleration structure (TLAS) for the
scene. Theia uses a different, meshlet-based layout. Only code that is identical for both
renderers lives in Harmonia, and Hyperion never includes Theia's code.

### Repository layout

| Path | Content |
|------|---------|
| `shaders/` | The ray-tracing shaders: `raygen.slang` (ray generation and path loop), `common.slang` (bindings and ray payload), `closesthit.slang` (surface shading; `closesthit_pf.slang` is its variant for `VK_KHR_ray_tracing_position_fetch`), `miss.slang`, `shadow_miss.slang`, `anyhit.slang` (shadow opacity), `intersection.slang` (analytic spheres), `texsampler.slang` (texture fetch), and the specular manifold sampling adapters `sms_scene.slang`, `sms_sample.slang`, `sms_prior_build.slang`, `sms_resolve.slang` |
| `src/hyperion/ShaderPaths.hpp` | File names of the compiled shader stages |
| `src/hyperion/scene/` | `Scene`: the GPU scene layout (index buffers, instances, emissive-triangle list) and the TLAS |
| `src/hyperion/renderer/` | `PathTracer` (per-frame recording, manifold-sampling queue), `ShaderBindingTable`, `SamplingPlan` (how `--spp` maps to frames) |
| `src/demo/` | `Application`, the host subclass that parses the command line and wires scene, camera and animation to the renderer, and `main.cpp` |
| `tests/` | Unit, module and integration tests, and an SDL/Vulkan start-up diagnostic |
| `tools/` | Checks and helper scripts, see [tools/README.md](tools/README.md) |

Hyperion's classes live in `namespace hyperion`; shared classes live in `namespace harmonia`.

---

## Features

- Vulkan 1.4 ray-tracing pipeline with ray-generation, closest-hit, miss and intersection
  shaders, dispatched indirectly through `vkCmdTraceRaysIndirect2KHR`.
- Unidirectional path tracing with configurable bounce depth and sample count, converging
  without bias.
- Next-event estimation for emissive triangles, emissive analytic spheres and the environment
  map, with multiple importance sampling between environment sampling and BSDF sampling.
- OpenPBR Surface v1.1.2 through the shared Harmonia BSDF: thin-film iridescence, dispersion,
  coat, fuzz, and subsurface and transmission rendered as a volumetric random walk.
- Image-based lighting with importance sampling of the environment map.
- Analytic spheres traced by a procedural intersection shader.
- Cut-out opacity with matching transparent shadows; opacity micromaps
  (`VK_EXT_opacity_micromap`) when the device supports them.
- Specular manifold sampling for caustics of planar mirrors and closed smooth glass.
- Headless capture to a scene-referred EXR plus a tonemapped PNG; interactive window with
  progressive accumulation.
- Animation: per-frame deformation (skinning, morph targets), TLAS refit, a camera bound to an
  animated node, and sequence rendering.

---

## How the path tracer works

### Ray generation and the path loop

For every pixel, the ray-generation shader (`shaders/raygen.slang`) traces
`samplesPerPixel` samples. Each sample does the following:

1. It picks a jittered position inside the pixel and generates a primary ray through it (see
   [Camera model](#camera-model)).
2. It follows the path in a loop. Each iteration traces one ray; the closest-hit or miss shader
   updates the path state carried in the ray payload (`shaders/common.slang`): accumulated
   radiance, throughput, the next ray, the random-number state and the medium state.
3. It multiplies the radiance by the camera exposure, limits the sample at zero from below,
   replaces a non-finite value by zero and adds the sample to the pixel sum.

The loop ends when the path terminates, when the maximum number of surface interactions
(`--depth`) is reached, or when a path has traced `depth + 256` rays. The extra 256 leave room
for medium segments and opacity pass-throughs, which do not count as surface interactions. The
ray-tracing pipeline has a constant recursion depth of 2 (ray-generation → hit shader → shadow
ray → any-hit or shadow-miss shader); the path itself is iterated in the ray-generation shader,
so the bounce depth does not change the pipeline.

The depth counts surface interactions, starting with the camera ray's first hit. Depth 1 gives
direct lighting only; each further level adds one bounce of indirect light.

Random numbers come from an Owen-scrambled Sobol' sequence
(`Harmonia/shaders/rng.slang`): the scramble key depends on the pixel and on `--rng-seed`, and
the sample index is `frameIndex × samplesPerPixel + sample`. The first two random numbers of a
sample are always the pixel jitter and the next two the lens position (drawn only when the lens
radius is above zero), so the same seed always produces the same samples.

The shader writes the mean of its samples to the HDR target and keeps no history. Harmonia's
accumulation pass is the only stage that blends frames; blending twice would leave a permanent
variance floor.

### Camera model

The camera comes from the scene's `[camera]` table, assembled by Harmonia's `CameraAssembly`.

| Property | Source | Behavior |
|----------|--------|----------|
| Projection | `vertical_field_of_view` in the camera table or camera preset | Vertical field of view in degrees; 39.1° when the scene sets none. Near and far planes are derived from the camera-to-target distance. |
| Exposure | `ev100` in the camera table or preset | Every sample's radiance is multiplied by `1 / (1.2 × 2^EV100)`. Without `ev100` the default physical camera applies: f/8, 1/100 s, ISO 100. |
| Thin lens | none | The shader samples a point on a disk of radius `lensRadius` and aims the ray at the focal point at distance `focusDistance`. The scene camera assembly sets the lens radius to 0 and the focus distance to the camera-to-target distance, so scenes render with a pinhole camera. The lens code is in place but no scene key sets an aperture. |

### Direct lighting

At every surface hit the estimator samples the lights explicitly instead of waiting for a path
to reach them by chance (next-event estimation, NEE):

| Light | Sampling | Density |
|-------|----------|---------|
| Emissive triangles | One triangle is selected proportionally to its power (area × luminance) through a cumulative distribution that `hyperion::Scene` builds; a point on it is drawn uniformly with Shirley's square-root folding of the barycentric coordinates. | The selection probability is folded into the area density, which is converted to a solid-angle density. |
| Emissive analytic spheres | A direction is drawn uniformly in the cone the sphere subtends. | The selection probability is multiplied into the uniform cone density. |
| Environment map | A direction is drawn from the 256×128 separable CDF that Harmonia builds from the panorama's luminance weighted by sin(θ). | The CDF density. |

A shadow ray tests the visibility of each sample. It returns the transmittance: 0 when blocked,
1 when free, and the product ∏(1 − α) over cut-out surfaces it passes (see
[Opacity and cut-outs](#opacity-and-cut-outs)).

Multiple importance sampling (MIS) with the balance heuristic combines the two ways of reaching
the environment: the light-sampled contribution is weighted `envPdf / (envPdf + bsdfPdf)`, and a
path that escapes to the sky after a BSDF sample is weighted `bsdfPdf / (bsdfPdf + envPdf)`
(`shaders/miss.slang`). Emissive meshes are handled by NEE alone: the emission of an emitter hit
directly by a path is added only for camera rays and after specular (delta) bounces, where NEE
cannot sample the light.

### Surface shading

At a surface hit, `shaders/closesthit.slang` rebuilds the hit (position, raw geometric normal
from the winding, shading normal, tangent, UV), applies the OpenPBR texture maps and calls
Harmonia's `integrateSurface` in `Harmonia/shaders/path_integrator.slang`, the surface
estimator that Theia uses as well. The texture maps (`map_base_color`, `map_normal`, `map_orm`
with roughness in G and metalness in B, `map_emission_color`, `map_opacity`) are read through a
bindless array of combined image samplers (set 1, binding 4, up to 1024 entries). The
estimator traces its shadow rays through the `ITracer` interface; Hyperion's implementation in
the same shader traces them with `TraceRay` against the shadow hit groups. In order, the
estimator:

1. terminates the path at an emitter, adding its emission (see
   [Direct lighting](#direct-lighting) for when the emission is counted);
2. passes the ray through with probability 1 − α (OpenPBR opacity), a null interaction that does
   not consume a bounce;
3. samples the emissive lights and the environment (NEE);
4. samples the BSDF for the continuation direction and multiplies the throughput by the sample
   weight, attenuated by the shadow-terminator factor of Chiang et al. (2019);
5. applies Russian roulette from depth 3.

The BSDF itself (`evalBSDF`, `pdfBSDF`, `sampleBSDF`) is the shared OpenPBR implementation in
`Harmonia/shaders/bsdf_shared.slang`; the material model is documented once in the
[Harmonia README](https://github.com/McNopper/Harmonia/blob/main/README.md#material-model--openpbr-surface-v112).
The OpenPBR conformance tests live in this repository: `tests/unit/test_bsdf.cpp` mirrors the
shared BSDF formulas in C++ and checks them against Monte-Carlo integration, reciprocity and
energy conservation. A change to the shared BSDF is verified there first.

### Volumetric subsurface and transmission

Subsurface scattering and transmission through a medium are simulated as a random walk inside
the object, not with a diffusion approximation. When a BSDF sample makes the path enter a
medium, every following ray is a medium segment until the path leaves the object:

- The ray is limited to a sampled free-flight distance.
- If it reaches that distance without hitting geometry, the miss shader (`shaders/miss.slang`)
  performs a scattering event: a Henyey–Greenstein phase function whose mean cosine is
  `subsurface_scatter_anisotropy`, with Russian roulette on the walk's throughput.
- If it hits geometry, the closest-hit shader resolves a boundary crossing: Beer–Lambert
  transmittance for the distance travelled, then a Fresnel-weighted choice between leaving the
  object (refraction with the inverted relative IOR) and internal reflection, including total
  internal reflection.

The per-event mathematics is shared with Theia through `Harmonia/shaders/medium_walk.slang`.
Extinction is per color channel. The free-flight distance is sampled on one channel chosen
proportionally to its extinction, and the other channels are reweighted (a hero-wavelength
estimator). A medium that only absorbs (single-scattering albedo of zero) skips the walk
altogether and receives the exact deterministic Beer–Lambert transmittance at the boundary,
which adds no variance.

### Opacity and cut-outs

OpenPBR opacity α (`geometry_opacity`, multiplied by the `map_opacity` texture when present) is
resolved in two ways:

- Camera and indirect rays pass through with probability 1 − α. This is a null interaction: the
  ray continues unchanged and the path does not lose a bounce.
- Shadow rays carry a dedicated any-hit shader (`shaders/anyhit.slang`) that multiplies the
  transmittance by (1 − α) at each crossing and ends the search at a fully opaque hit, so a
  cut-out casts a cut-out shadow.

Only the two shadow hit groups carry an any-hit shader. When the device supports
`VK_EXT_opacity_micromap`, a baked micromap lets the traversal resolve fully opaque and fully
transparent micro-triangles without running the shader; the result is the same. The scene
`shaderball_checker` shows the same opacity mask with and without a micromap.

### Specular manifold sampling

Light that reaches a diffuse surface through a chain of specular interactions (a caustic under
glass, a lamp seen through a mirror) is found rarely by plain path tracing. Specular manifold
sampling (SMS, Zeltner et al. 2020) starts the search from the diffuse receiver instead: it
samples a light point and solves for the specular vertices that connect the receiver to that
point. The probability of finding such a connection is estimated by repeating the search from
the same receiver with new random numbers and counting the repeats that find the same connection.

Hyperion uses the shared implementation from Harmonia
([Harmonia README, Specular manifold sampling](https://github.com/McNopper/Harmonia/blob/main/README.md#specular-manifold-sampling))
in its partitioned form with per-frame priors (Hong et al. 2025): the search is restricted to a
3×3 neighborhood of tiles in primary space (`smsPartitionResolution` tiles per axis, default
16, valid values 1 to 32; 1 is the unpartitioned sampling), and 128 pilot walks per 16×16 pixel
block select the tiles that repeat trials may use. A uniform component of the prior keeps the
coverage complete. Both settings are fields of `PathTracer::Config`: with `smsPriorEnabled`
switched off, every tile is sampled uniformly, which serves as the control for a comparison.

Division of work:

- The ray-tracing shaders only prepare it. When a camera path reaches the first opaque,
  non-specular surface after a purely specular prefix, the closest-hit shader stores the receiver
  in a work record. The path itself continues with ordinary path tracing.
- A compute pass (`shaders/sms_sample.slang`, which includes
  `Harmonia/shaders/sms_compute.slang`) runs the trials.
- A resolve pass (`shaders/sms_resolve.slang`) adds the resulting radiance to the pixel after
  validating every work record. A failed validation fails the whole frame instead of
  accumulating a partial result.

Visibility inside the trials comes from an opaque ray query against the scene's acceleration
structure, analytic spheres included (`shaders/sms_scene.slang`). A hit whose shading data
cannot be reconstructed never counts as "not occluded".

Supported connections are the reflection at, and the transmission through, a smooth untextured
glass surface (specular roughness below 0.05, no coat, no fuzz layer, no dispersion) whose
interior is either without volumetric depth or a pure absorber. The fixtures in
`tests/module/test_sms_frame.cpp` verify the energy of a planar mirror and of flat closed glass
against independent references. Everything else stays with ordinary path tracing: scattering
media, dispersion, rough specular surfaces, longer chains, and receivers after a diffuse bounce.

`--sms-mode off` disables the sampling: the shaders collect no work and the image is the plain
path-traced result. This is the reference mode for comparisons that involve manifold sampling.

### Analytic spheres

A scene can declare procedural spheres (`type = "sphere"` in a mesh entry). They are traced by
an intersection shader (`shaders/intersection.slang`) in object space, so an instance transform
can turn them into ellipsoids, and they take the same materials as meshes. An emissive sphere is
sampled by NEE like a triangle light as long as its transform is a uniform scale; an ellipsoid
emitter is skipped with a warning.

### Animation

One animation clock in the host (`App::animationClock`) drives every animated consumer, so
they cannot disagree: the scene's animated transforms, the vertex deformation (skinning and morph
targets, through Harmonia's `DeformPass`), the TLAS update and a camera bound to a scene node.
`Application::record` in `src/demo/Application.cpp` applies them in this order whenever the
animation time changes: stamp the transforms, deform the vertices, update the TLAS (refit for
animated scenes, rebuild otherwise; the policy is Harmonia's), and restart the accumulation,
because the samples accumulated so far belong to an earlier animation state. An animated scene
therefore needs one accumulation per animation time; see
[Producing a reference render](#producing-a-reference-render).

---

## GPU-driven rendering

`PathTracer::render()` in `src/hyperion/renderer/PathTracer.cpp` records the commands of one
logical frame: it writes the small camera uniform block, binds the ray-tracing pipeline and the
descriptor buffers, pushes the frame constants, dispatches the rays and starts the
manifold-sampling queue. It does not wait for the GPU and reads nothing back; whether the frame
is finished is queried separately through `frameReady()` after the submission has completed.

| Mechanism | Detail |
|-----------|--------|
| Indirect ray dispatch | `vkCmdTraceRaysIndirect2KHR` (`VK_KHR_ray_tracing_maintenance1`, a hard requirement: device selection fails without it). The indirect command holds the shader-binding-table addresses and the render dimensions; `PathTracer::updateIndirectBuffer()` writes it at creation and on resize. |
| Descriptor buffer | `VK_EXT_descriptor_buffer` (a hard requirement): descriptors are written into GPU buffers and bound with `vkCmdBindDescriptorBuffersEXT`. |
| Acceleration structures | Built and updated with device-side commands only (`vkCmdBuildAccelerationStructuresKHR`); the host-side build functions are not used. Creation goes through `vkCreateAccelerationStructure2KHR` (`VK_KHR_device_address_commands`) only. |
| `VK_KHR_ray_tracing_position_fetch` (optional) | The closest-hit shader reads triangle vertex positions from the hardware (`closesthit_pf.spv`) and transforms them by the instance matrix. |
| `VK_EXT_ray_tracing_invocation_reorder` (optional) | The ray dispatch asks for invocation reordering. |
| `VK_EXT_opacity_micromap` (optional) | The pipeline is created so that it can traverse micromapped geometry. |

Compiled pipelines are cached as driver binaries by Harmonia (`VK_KHR_pipeline_binary`): the
cache is on by default under `<exe-dir>/pipeline-cache`, `HARMONIA_PIPELINE_BINARY_CACHE`
overrides the directory, and the values `off` and `0` disable it. A pipeline for which the driver
returns no usable key is compiled from its shaders instead. The cache changes start-up time
only, never the image. Details:
[Harmonia README, Pipeline binary cache](https://github.com/McNopper/Harmonia/blob/main/README.md#pipeline-binary-cache).

---

## Topics documented in Harmonia and Aether

Hyperion uses these topics unchanged. They are described once in the repository that owns them.

| Topic | Summary | Reference |
|-------|---------|-----------|
| Material model | OpenPBR Surface v1.1.2 is implemented once in `Harmonia/shaders/bsdf_shared.slang` and used identically by both renderers: layer stack, conductor and dielectric Fresnel, thin-film iridescence, fuzz, the dielectric sidedness rule and the transmission tint semantics. | [Harmonia README, Material model](https://github.com/McNopper/Harmonia/blob/main/README.md#material-model--openpbr-surface-v112) |
| Surface estimator and shared shaders | Next-event estimation, BSDF sampling and Russian roulette in `Harmonia/shaders/path_integrator.slang`; the manifold-sampling solver in the shared modules of `Harmonia/shaders/`, included through `Harmonia/shaders/sms_compute.slang`. | [Harmonia README, Shared shaders](https://github.com/McNopper/Harmonia/blob/main/README.md#shared-shaders) |
| Common command-line flags | Harmonia's `CliParser` parses the flags shared by both renderers (scene selection, output, resolution, validation, random seed, sequence capture). | [Harmonia README, Command-line flags](https://github.com/McNopper/Harmonia/blob/main/README.md#command-line-flags) |
| Frame processing and capture contract | The host submits a frame, drains the renderer's continuations until the frame is ready, then runs accumulation and tone mapping; a capture under `--output` is the raw scene-referred result without the denoiser. | [Harmonia README, How a frame is rendered](https://github.com/McNopper/Harmonia/blob/main/README.md#how-a-frame-is-rendered), [Harmonia ARCHITECTURE.md, Logical frame completion](https://github.com/McNopper/Harmonia/blob/main/ARCHITECTURE.md#logical-frame-completion) |
| Parity methodology and thresholds | A Theia capture is compared with a Hyperion reference EXR; the metrics, their limits and the comparison tools belong to Harmonia. | [Harmonia README, Parity methodology](https://github.com/McNopper/Harmonia/blob/main/README.md#parity-methodology), `Harmonia/tools/validation_manifest.toml` |
| Pipeline binary cache | Driver pipeline binaries persisted across runs, keyed by the driver's pipeline key. | [Harmonia README, Pipeline binary cache](https://github.com/McNopper/Harmonia/blob/main/README.md#pipeline-binary-cache) |
| Scene, material and camera format | TOML scene, material and preset files and OBJ geometry, parsed by Aether. | [Aether README, File format reference](https://github.com/McNopper/Aether/blob/main/README.md#file-format-reference) |

---

## Scenes

Scenes are TOML files parsed by Aether: a `<name>.scene.toml` references OpenPBR material
libraries (`<name>.materials.toml`), OBJ geometry and presets for the camera, render settings and
tonemapper. The format is documented in the
[Aether README](https://github.com/McNopper/Aether/blob/main/README.md#file-format-reference).
The following points matter for every Hyperion scene:

| Point | Detail |
|-------|--------|
| Camera placement is mandatory | The `[camera]` table needs a translation and a rotation, inline or through a camera preset that carries both. The shared scene loader rejects a scene without them and names the missing part; there is no default view. A preset that only sets the field of view and `ev100` does not satisfy the requirement. |
| Sample count and depth | `[render]` `samples_per_pixel` and `max_depth`, inline or through a render preset, are the defaults for `--spp` and `--depth`. |
| Environment lighting | `[render]` `environment_map` names an equirectangular OpenEXR panorama; `environment_unit_nits` scales it (cd/m² per EXR unit). |
| Working color space | `[render]` `working_color_space` selects the scene-referred space, `lin_rec2020_scene` (default) or `lin_rec709_scene`. Assets are converted on load. |
| Geometry | OBJ files contribute geometry only; materials are assigned in the scene file. |
| OBJ winding | Faces must be wound outward (counter-clockwise seen from outside). Emissive-triangle normals derive from the winding, and back-facing emitters are skipped by NEE, so an inward-wound object renders black. |
| Assets directory | Hyperion reads scenes from the `Aether/assets/` directory, whose path is compiled into the executable as `HYPERION_ASSETS_DIR`. |

The assets directory is the Aether checkout that CMake resolved: by default the `FetchContent`
clone under `build/_deps/aether-src/`, otherwise the directory given to
`FETCHCONTENT_SOURCE_DIR_AETHER`. Editing a scene in a different Aether working tree has no
effect on a render until that tree is the one the build uses. The symptom is two "different"
renders that are byte-identical.

### Color and output

Rendering is scene-referred: every value in the HDR target is linear radiance in the working
color space. Tone mapping (AgX, ACES, Reinhard or Hable, selected per scene in the `[tonemap]`
table) and the display output (SDR, HDR10 or scRGB, negotiated with the swapchain) are shared
Harmonia stages.

| Output | Content |
|--------|---------|
| EXR | The scene-referred, untonemapped, accumulated frame, tagged with the working color space. This is the file for every comparison. |
| PNG | The same frame after the GPU tone mapper that the interactive window uses. |

---

## Building

| Requirement | Notes |
|-------------|-------|
| Vulkan SDK 1.4 | `VULKAN_SDK` must be set. The SDK provides the headers, `slangc`, SDL3, volk and VMA. |
| CMake 3.28 or newer, Ninja | |
| clang-cl (Windows) or g++ (Linux) | The code is C++23. Other compilers are rejected by `cmake/Platform.cmake`. |
| vcpkg | Its toolchain file provides tomlplusplus and OpenImageIO. OpenImageIO is built from source on the first configure, which takes several minutes; later configures reuse the binary cache. |
| A ray-tracing capable GPU and driver | Vulkan 1.4 with ray tracing, `VK_KHR_ray_tracing_maintenance1`, `VK_EXT_descriptor_buffer` and `VK_KHR_device_address_commands`. |

Windows, with the toolchain file of the vcpkg installation:

```powershell
cmake -S . -B build -G Ninja `
      -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl `
      -DCMAKE_TOOLCHAIN_FILE="<vcpkg-root>/scripts/buildsystems/vcpkg.cmake"
cmake --build build
```

The preset in `CMakePresets.json` is equivalent when `VCPKG_ROOT` is set:
`cmake --preset win`, then `cmake --build --preset win`.

Linux:

```bash
export VULKAN_SDK=<extracted-sdk>/<version>/x86_64
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_TOOLCHAIN_FILE="$VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build
ctest --test-dir build --output-on-failure
```

The Linux SDK tarball ships without SDL3. Install the distribution's `libsdl3-dev` package and
link it into the SDK layout (the SDL3 header folder under the SDK `include` directory, and
`libSDL3.so` under its `lib` directory), as the continuous integration workflow does.

By default CMake downloads Aether and Harmonia at the release tags pinned in `CMakeLists.txt`.
To build against local checkouts instead, pass
`-DFETCHCONTENT_SOURCE_DIR_AETHER=<path> -DFETCHCONTENT_SOURCE_DIR_HARMONIA=<path> -DFETCHCONTENT_SOURCE_DIR_SLANG_MATH=<path>`.

The entry shaders in `shaders/`, which import the shared modules from `Harmonia/shaders/`, are
compiled at build time by Harmonia's `compile_slang_shaders` CMake function into
`build/shaders/*.spv`, next to Harmonia's own shaders. The executable is `build/hyperion.exe`
on Windows and `build/hyperion` on Linux and loads its shaders from the shader directory
compiled into it (`HYPERION_SHADER_DIR`).

---

## Running

```powershell
build/hyperion.exe shaderball_base                                       # interactive window
build/hyperion.exe --scene cornell_classic --output cornell.png          # headless, PNG only
build/hyperion.exe --scene cornell_classic --output cornell.exr          # headless, EXR + PNG
build/hyperion.exe --scene dragon_teapot --spp 256 --output ref.exr       # clean reference
```

### Command-line flags

Hyperion adds three flags to the common flags of the Harmonia host.

| Flag | Default | Meaning |
|------|---------|---------|
| `--spp <n>` | `samples_per_pixel` of the scene's `[render]` table; 4 when the scene sets none | Samples per pixel. In a capture, N single-sample frames are accumulated (N is the total sample count, not N frames of N samples). In the interactive window, N samples are traced per dispatch. A value below 1 is treated as 1. |
| `--depth <n>` | `max_depth` of the scene's `[render]` table; 8 when the scene sets none | Maximum number of surface interactions per path. |
| `--sms-mode <off\|local>` | `local` | `off` renders with ordinary path tracing only, without specular manifold sampling (the reference mode for manifold-sampling comparisons). `local` enables manifold sampling. Reservoir reuse exists only in Theia; any other value is rejected. |

`--spp` and `--depth` accept both `--spp 256` and `--spp=256`. A value given on the command
line overrides the scene; otherwise the scene's values apply. An unknown argument aborts the
start with exit code 2.

The common flags parsed by Harmonia that matter for Hyperion:

| Flag | Default | Meaning |
|------|---------|---------|
| `--scene <name>` / `-s` | `cornell_classic` | Scene name or path. A bare name resolves against the assets directory, and a scene can also be given as a bare argument. |
| `--output <file>` / `-o` | none | Headless mode: the window stays hidden, the scene is rendered, the files are written and the program exits. There is no `--offscreen` flag. |
| `--width <n>`, `--height <n>` | 1024, 768 | Render resolution in pixels. |
| `--rng-seed <n>` | 305419896 | Base seed of the random-number sequence. |
| `--validation` / `--no-validation` | enabled | Vulkan validation layers. Validation stays on unless it is switched off explicitly. |
| `--sequence-frames <N>` | 0 | With `--output`: renders N animation frames to disk instead of one accumulated image. Needed for a clean capture of an animated scene. |

The complete list, including the animation and presentation flags, is in the
[Harmonia README](https://github.com/McNopper/Harmonia/blob/main/README.md#command-line-flags).

### Interactive window

Without `--output`, Hyperion opens a 1024×768 window and accumulates progressively, so the
displayed image converges to the same result as a capture. The `S` key saves the image that the
scene stages currently produce as `hyperion_<frame>.exr` in the working directory; with the
denoiser enabled this is the denoised image, and `--no-postfx` gives the raw accumulated image.

---

## Producing a reference render

A reference is a Hyperion EXR that another image is compared with.

```powershell
build/hyperion.exe --scene <name> --spp 256 --width 320 --height 240 --output <dir>/<name>.exr
```

| Rule | Reason |
|------|--------|
| Compare EXR files, never PNG files. | The EXR is the scene-referred linear result; the PNG is tone mapped. |
| Use 256 samples per pixel at 320×240 for parity references. | This is the budget of `Harmonia/tools/validation_manifest.toml`. The comparison thresholds are calibrated for it and must be re-measured if the budget changes. |
| Do not use a low-sample reference. | The difference between a noisy reference and a candidate is mostly the reference's own Monte-Carlo noise. Under image-based lighting the scene presets range from 64 to 256 samples per pixel, and 64 is not enough: pass `--spp 256` explicitly. The gallery's 64 samples per pixel are not a reference. |
| Set `--rng-seed` for a reproducible reference. | With the same seed and settings a capture repeats sample for sample. |
| Capture of an animated scene: `--sequence-frames <N>`. | Without it the accumulation restarts at every sample because the animation advances, and the capture is a single-sample image of the last animation time (a warning says so). `--sequence-frames <N>` writes N accumulated images, one per animation frame; `--sequence-frames 1 --frame <k>` writes the single image at animation frame k. |
| One GPU job at a time. | Renders, gallery generation and comparison runs each saturate the GPU; two concurrent jobs are both slower. |

A capture runs at below-normal CPU priority so that the desktop stays usable while it renders.
That does not make two concurrent captures acceptable: the GPU is the limit.

Harmonia's `--offscreen-frames` has no effect in Hyperion: the number of accumulated frames of
a capture is always the sample count.

A capture is the raw estimator result: the accumulation stage blends the single-sample frames,
the denoiser is off, and a scene that enables the denoiser stage for a capture is refused. A
capture with more than one sample per pixel needs the accumulation stage; a scene that disables
it is refused as well.

For a comparison with manifold sampling involved, add `--sms-mode off` and raise the sample
count: the reference images of the manifold-sampling comparison were rendered with 4096 samples
per pixel and split seeds, with an uncertainty of at most 0.5%.

The comparison itself (rendering both renderers, computing the metrics, applying the limits)
is done with Harmonia's tools and is described in the
[Harmonia README](https://github.com/McNopper/Harmonia/blob/main/README.md#parity-methodology).
To check a change for Vulkan validation messages, run Harmonia's
`Harmonia/tools/check_vulkan_validation.py`; it fails on any validation message and on any
warning or error line of the application log.

### Sequence capture of an animated scene

`--sequence-frames <N>` together with `--output` renders N animation frames to disk.

| Setting | Effect |
|---------|--------|
| `--sequence-frames <N>` | Output frame k shows the scene at animation frame k (time k / frame rate; the rate comes from the scene's animation clock, 30 per second by default). Each output frame is an independent accumulation of all `--spp` samples at that time. The files are written as `<stem>_0000.exr`, `<stem>_0000.png`, `<stem>_0001.exr`, and so on. |
| `--frame <k>` | Starts the animation at frame k; output frame j shows animation frame k + j. |
| `--capture-every-ms <ms>`, `--start-seconds <s>` | Samples the animation time every ms milliseconds, starting at s seconds, instead of the frame grid. `--capture-every-ms` requires `--sequence-frames`. |
| `--shutter <f>`, `--time-samples <n>` | Motion blur. `--shutter` is the fraction of the frame interval during which the shutter is open and `--time-samples` the number of times inside it that are rendered. Each of the `--spp` samples of an output frame is placed in one of n equal parts of the open shutter, at a jittered position, so the accumulated image is the integral over the exposure. `--spp` must be at least n. |

`Harmonia/tools/frames_to_video.py` encodes such a sequence into an H.264 video.

A shuttered Hyperion frame is the exact mean of its samples, so it is the reference image for
motion blur. It is compared with a Theia frame rendered with the same flags and the same
`--rng-seed`, because the jittered position of each time sample depends on the seed. The Aether
scene `motion_blur` is made for this comparison: at 320×240 with 512 samples, a shuttered frame
against a shuttered Theia frame passes the parity limits (SSIM 0.82, luminance-histogram
correlation 0.81), and a sharp frame against a shuttered one does not.

---

## Test scenes

Scenes live in the `Aether/assets/` directory. A bare scene name selects `<name>.scene.toml`.

| Group | Scenes | Render preset | Use |
|-------|--------|---------------|-----|
| Cornell box | `cornell_classic` (the default scene), `cornell_spheres`, `cornell_suzanne`, `cornell_textured_cube`, `cornell_empty`, `cornell_classic_rec709` | 64 samples per pixel, 8 bounces, direct light only | Quick iteration. `cornell_classic_rec709` renders `cornell_classic` in the Rec.709 working color space. |
| Material fixtures | `openpbr_metals`, `openpbr_dielectrics`, `openpbr_coat`, `openpbr_fuzz`, `openpbr_specular`, `openpbr_organics`, `openpbr_thinfilm`, `openpbr_advanced`; `shaderball_base`, `shaderball_metal`, `shaderball_coat`, `shaderball_fuzz`, `shaderball_specular`, `shaderball_opacity`, `shaderball_subsurface`, `shaderball_thinfilm`, `shaderball_transmission`, `shaderball_emission`, `shaderball_checker` | 64 to 256 samples per pixel, image-based lighting | One material feature each. `shaderball_checker` shows the same opacity mask with and without an opacity micromap. |
| Showcases | `shader_ball`, `dragon_teapot`, `bunny_shaderball`, `camera_suzanne`, `ABeautifulGame` | 256 samples per pixel, 12 or 16 bounces | `ABeautifulGame` is by far the most expensive scene; use it for gallery renders only, not for iteration. |
| Animation | `camera_orbit`, `camera_static`, `skinned`, `skinned_pose`, `skinned_dqs`, `skinned_orbit` | | A camera bound to an animated node, a camera bound to a static node, linear-blend and dual-quaternion skinning, morph targets. |

---

## Tests

### Test suites

All suites use Google Test and need a GPU except the unit suite. Every test is registered with
`RUN_SERIAL`, so `ctest -j` cannot parallelize them.

```powershell
cd build; ctest --output-on-failure     # or: ctest --preset win
```

| Executable | Sources | Covers |
|------------|---------|--------|
| `hyperion_unit_tests` | `tests/unit/test_bsdf.cpp`, `tests/unit/test_sampling_plan.cpp` | OpenPBR conformance of the shared BSDF (mirrored in C++): energy conservation, reciprocity, sampler and density consistency, thin film, sheen, medium exit with total internal reflection. The sample-count plan. |
| `hyperion_module_tests` | `tests/module/` | Scene build (BLAS and TLAS for meshes, spheres and instances), animated TLAS refit, shader-binding-table compliance, frame sampling (the ray-generation shader writes only the current frame; the shared accumulation is the only history owner), the manifold-sampling scene reader and its end-to-end fixtures. |
| `hyperion_integration_tests` | `tests/integration/` | A start-up smoke test, a Cornell-box render that must be non-black, and the capture sampling contract (sample budget, reproducibility per seed, refusal of a multi-sample capture without accumulation). |
| `hyperion_diag` | `tests/diag/test_sdl_vulkan_init.cpp` | Not a test: a stand-alone program that initializes the Vulkan loader, SDL3, a hidden window, a Vulkan instance and a surface, and enumerates the physical devices, printing a `DIAG:` line for every step (`all steps passed` at the end). It locates a start-up crash without the renderer. |

A single test can be run directly, for example
`build/tests/hyperion_module_tests.exe --gtest_filter=FrameSamplingFixture.*`.

### Static analysis

`ctest -LE analysis` runs the test suites only. The tests labelled `analysis` are checks over
the sources; `ctest -L analysis` runs them alone.

| Check | Invocation | What it reports |
|-------|-------------|-----------------|
| clang-tidy | `python tools/check_tidy.py` (ctest `test_tidy`) | Findings of the checks enabled in `.clang-tidy`, in parallel over the translation units of `build/compile_commands.json`. Findings of the categories `clang-diagnostic`, `clang-analyzer` and `bugprone` fail the run; `modernize`, `performance` and `portability` findings are listed only. |
| Architecture contract | `python tools/check_arch.py` (ctest `test_arch`) | Whether the include graph of `src/` matches [ARCHITECTURE.md](ARCHITECTURE.md). The test is registered only on machines where the check-arch engine is installed in the user profile. |
| Dead code | `python tools/check_dead.py` (ctest `test_dead`) | Headers that nobody includes, sources that are neither compiled nor referenced in a CMake file, and `FetchContent` declarations that are never used. The test is registered only where the check-dead engine is installed in the user profile. |
| Formatting | `cmake --build build --target format-check` | clang-format differences (`format` applies them). |

The CMake targets `verify` (build and test) and `verify-full` (verify, `format-check`, `tidy`
and `cppcheck`) run the checks together. `cppcheck-strict` and `cppcheck-xml` are the exhaustive
and the machine-readable cppcheck profiles.

Two further configurations exist for Clang and GCC builds only:

| Configuration | Command | Purpose |
|---------------|---------|---------|
| Sanitizers | `-DHYPERION_SANITIZER=address`, `undefined` or `thread` | Run the suites under a sanitizer. |
| Coverage | `-DHYPERION_COVERAGE=ON` in a separate build directory such as `build/coverage`, then `python tools/coverage.py` | Source coverage of `src/` under the test suites. Not part of CI, because CI has no GPU. |

### Continuous integration

`.github/workflows/build.yml` runs on pushes to `main`, on pull requests that target `main`,
and on manual dispatch.

| Job | Runner | Content |
|-----|--------|---------|
| `lint` | `ubuntu-latest` | Blocking: clang-format 22.1.3 over the C++ sources of `src/` and `tests/`; cppcheck over `src/` (no `error` severity allowed); ruff and bandit over `tools/`; actionlint and yamllint over the workflows. |
| `build` | `windows-latest` and `ubuntu-26.04` | Configures (clang-cl on Windows, g++ on Linux) and builds against the `main` branches of the sibling repositories; the two manifold-sampling development branches use the matching Harmonia branch. The test suites are not run: they need a GPU and stay a local activity. |

A change to the Python or workflow files should pass the same lint commands locally before it
is pushed.

---

## Conventions

- The estimator stays unbiased. No clamp, special case or scene-specific behavior in the render
  path; the display path may filter, the captured image may not.
- The shared accumulation pass is the only owner of the frame history. A shader writes the
  current dispatch only.
- Defects are fixed in the code that owns them. A tolerance is never loosened to make a test
  pass, and a defect of a shared stage is fixed in Harmonia, not worked around here.
- Code shared by both renderers belongs in Harmonia. Hyperion never depends on Theia; an include
  of a `theia/` path is an architecture violation.
- Builds are warning-clean: a compiler warning fails the build (`/W4 /WX` with clang-cl,
  `-Wall -Wextra -Werror -Wpedantic` with Clang and GCC). A Vulkan validation message is a
  defect.
- Vulkan usage is GPU-driven and cross-vendor: the latest stable Vulkan with core and `KHR`/`EXT`
  extensions only, no `VK_NV_*`, `VK_AMD_*` or `VK_INTEL_*` extension, and no fallback path
  around a missing capability.
- The working color space is scene-referred and linear (`lin_rec2020_scene` or
  `lin_rec709_scene`).
- One GPU job at a time: renders, gallery generation and comparison runs do not overlap, and
  test suites of different repositories are not run concurrently.
- Changes are committed in the repository they belong to and are pushed only on request. The
  release order follows the dependency direction: slang-math, Aether, Harmonia, Hyperion; the
  Harmonia and Aether pins in `CMakeLists.txt` move in the release commit.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| [Aether](https://github.com/McNopper/Aether) | Scene and material file formats (`.scene.toml`, `.materials.toml`, OBJ) and the assets; CPU data without GPU dependencies |
| [Harmonia](https://github.com/McNopper/Harmonia) | Shared Vulkan foundation: application host, device and presentation, color management, tonemapping, accumulation, shared shaders and estimators |
| [Vulkan SDK](https://vulkan.lunarg.com/) | Ray-tracing API, Slang compiler, SDL3, volk, VMA |
| [slang-math](https://github.com/McNopper/slang-math) | Math library, pulled transitively through Harmonia |
| [OpenImageIO](https://openimageio.readthedocs.io/) | Image I/O: PNG and JPEG load, EXR load and save with chromaticities (via Harmonia; stb and OpenEXR come with it) |
| [Slang](https://shader-slang.com/) | Shader language |
| [Google Test](https://github.com/google/googletest) | Testing |

---

## References

The references of the shared BSDF, the volumetric walk, the color management and the denoiser
are in [Harmonia's README](https://github.com/McNopper/Harmonia/blob/main/README.md#references).
Below are the references specific to the path tracer.

| Resource | Relevance |
|----------|-----------|
| [Physically Based Rendering: From Theory To Implementation, 4th ed.](https://www.pbrt.org/) (Pharr, Jakob, Humphreys) | MIS balance heuristic (§13.4.3), area light NEE (§12.4), environment importance sampling with a separable CDF (§12.5) |
| [Shirley, Wang & Zimmerman — Monte Carlo Techniques for Direct Lighting Calculations (J. Graphics Tools 1(1), 1996)](https://doi.org/10.1080/10867651.1996.10487479) | Uniform area sampling of triangles with square-root folded barycentric coordinates; conversion between area and solid-angle density |
| [Zeltner, Georgiev & Jakob — Specular Manifold Sampling (SIGGRAPH 2020)](https://doi.org/10.1145/3386569.3392408) | Specular manifold sampling and its reciprocal-probability estimate |
| [Hong et al. — Partitioned SMS and ReSTIR (SIGGRAPH Asia 2025)](https://doi.org/10.1145/3757377.3763927) | Trust regions in primary space and priors for manifold sampling |
| [Vulkan Specification 1.4](https://registry.khronos.org/vulkan/specs/latest/html/) | `VK_KHR_ray_tracing_pipeline`, `VK_KHR_acceleration_structure`, descriptor indexing, `VK_KHR_ray_tracing_maintenance1` |
| [Slang Shading Language](https://shader-slang.com/) | `[raypayload]` attribute, `TraceRay`, Vulkan binding attributes |
| [Wavefront OBJ](http://paulbourke.net/dataformats/obj/) | Geometry-only OBJ import; material directives are ignored |
| [OpenPBR Surface Specification v1.1.2](https://academysoftwarefoundation.github.io/OpenPBR/) | Material model and parameter naming |
