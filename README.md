# Hyperion

Vulkan path-tracer for OpenPBR.

> *[Hyperion](https://en.wikipedia.org/wiki/Hyperion_(mythology)) — Titan of heavenly light, father of Helios, Selene and Eos.*

Hyperion is a GPU path tracer built entirely on Vulkan 1.4 KHR ray tracing.  
It implements the [OpenPBR Surface v1.1.1](https://academysoftwarefoundation.github.io/OpenPBR/) material model and outputs a physically correct, linearly encoded HDR frame every render call.

---

## Screenshots

| Cornell Box | Spheres | Suzanne |
|:-----------:|:-------:|:-------:|
| ![cornell_classic](screenshots/cornell_classic.png) | ![cornell_spheres](screenshots/cornell_spheres.png) | ![cornell_suzanne](screenshots/cornell_suzanne.png) |

| Metals | Dielectrics | Coat |
|:------:|:-----------:|:----:|
| ![openpbr_metals](screenshots/openpbr_metals.png) | ![openpbr_dielectrics](screenshots/openpbr_dielectrics.png) | ![openpbr_coat](screenshots/openpbr_coat.png) |

| Fuzz | Specular | Organics |
|:----:|:--------:|:--------:|
| ![openpbr_fuzz](screenshots/openpbr_fuzz.png) | ![openpbr_specular](screenshots/openpbr_specular.png) | ![openpbr_organics](screenshots/openpbr_organics.png) |

| Thin-film | Special Materials | Meadow IBL |
|:---------:|:-----------------:|:----------:|
| ![openpbr_thinfilm](screenshots/openpbr_thinfilm.png) | ![openpbr_special](screenshots/openpbr_special.png) | ![meadow_scene](screenshots/meadow_scene.png) |

| Textured Cube | Dragon & Teapot (IBL) | Advanced Transmission |
|:-------------:|:---------------------:|:---------------------:|
| ![textured_cube](screenshots/textured_cube.png) | ![dragon_teapot](screenshots/dragon_teapot.png) | ![openpbr_advanced](screenshots/openpbr_advanced.png) |

| A Beautiful Game (IBL) | Bunny + ShaderBall (IBL) | Blender Export |
|:----------------------:|:------------------------:|:------------------------:|
| ![ABeautifulGame](screenshots/ABeautifulGame.png) | ![bunny_shaderball](screenshots/bunny_shaderball.png) | ![camera_suzanne](screenshots/camera_suzanne.png) |

---

## Features

### Rendering
- Vulkan 1.4 KHR ray tracing pipeline (raygen / closest-hit / miss / intersection shaders)
- Unidirectional path tracing with configurable bounce depth and samples per pixel
- **Next Event Estimation (NEE)** with **Multiple Importance Sampling (MIS)** — balance heuristic combining BSDF pdf and light pdf; eliminates black-dot noise and halves required SPP
- **Emissive mesh area lights** — per-triangle direct sampling using Shirley's sqrt-folding barycentric coordinates; area-to-solid-angle PDF conversion
- **Environment map importance sampling** — 2D separable CDF (256×128) built from panorama luminance × sin(θ); MIS-weighted against BSDF paths in the miss shader; eliminates fireflies from bright suns and skies
- Analytic spheres via `VK_KHR_ray_tracing_pipeline` intersection shaders
- Image-based lighting (IBL) — equirectangular HDR panorama via `env_map`
- Firefly suppression (channel-average clamping with NaN guard)
- À trous wavelet denoiser pass
- Headless render mode with PNG + EXR output

### Material model — OpenPBR Surface v1.1.1
All parameters follow the [OpenPBR spec](https://academysoftwarefoundation.github.io/OpenPBR/) naming:

| Layer | Parameters |
|-------|-----------|
| Base | `base_weight`, `base_color`, `base_diffuse_roughness`, `base_metalness` |
| Specular | `specular_weight`, `specular_color`, `specular_ior`, `specular_roughness`, `specular_roughness_anisotropy` |
| Coat | `coat_weight`, `coat_color`, `coat_ior`, `coat_roughness`, `coat_darkening` |
| Fuzz | `fuzz_weight`, `fuzz_color`, `fuzz_roughness` |
| Emission | `emission_luminance`, `emission_color` |
| Thin-film | `thin_film_weight`, `thin_film_thickness`, `thin_film_ior` |
| Transmission | `transmission_weight`, `transmission_color`, `transmission_depth` |
| Subsurface | `subsurface_weight`, `subsurface_color`, `subsurface_radius`, `subsurface_radius_scale`, `subsurface_scatter_anisotropy` |
| Geometry | `geometry_opacity` |

Conductor reflectance uses the OpenPBR generalized-Schlick **F82-tint** model
(`base_color` = F0, `specular_color` = the 82° tint); specular/coat microfacets
use GGX with the spec's anisotropy remapping plus Turquin/Kulla-Conty multiple-scattering
energy compensation on both the reflection **and** the rough-transmission lobes (so rough
glass does not lose energy).

**Layer stacking** follows OpenPBR's directional-albedo coupling rather than a linear blend:
each lower layer is attenuated by the directional albedo of the layer above it
(`R_out = R_top + (1 − E_top)·R_base`). The coat darkens the substrate by `1 − coat·E_coat`
and the dielectric diffuse/subsurface base sits under the specular layer (`1 − E_spec`); both
are applied symmetrically in view/light so the stack stays reciprocal and energy-conserving.

**Thin-film iridescence** is the spec model — a faithful port of MaterialX `mx_fresnel_airy`
(Belcour & Barla 2017): a full s/p-polarized Airy summation with the spectral Gaussian
sensitivity. For metals it uses the true **complex-IOR conductor phase** (with `(n,k)`
recovered from `base_color` + `specular_color` via the Gulbrandsen 2014 artist-friendly
mapping), so anodized metals show vivid, physically-correct interference colour; dielectric
bases use the Schlick interface, and the two are blended by `base_metalness`.

**Fuzz/sheen** is the OpenPBR spec model — a faithful port of MaterialX's Zeltner et al. 2022
"Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines". The LTC coefficients
and the sheen directional albedo are closed-form analytic fits (no lookup table), and that
directional albedo also drives the physically-correct, view-dependent darkening of the layers
beneath the fuzz.

**Subsurface** (bulk, non-thin-walled) is a **real volumetric random walk**, not a diffusion
or tinted-diffuse approximation: light refracts through the dielectric interface (Fresnel-gated),
takes an exponential free-flight walk with Henyey-Greenstein phase scattering
(`subsurface_scatter_anisotropy` = the phase mean cosine), and exits through the interface with
Fresnel-gated transmission / total internal reflection. Extinction is **chromatic (per-channel)** —
derived from `subsurface_radius` × `subsurface_radius_scale`, so the default `(1, 0.5, 0.25)` scale
gives the characteristic red-shifted subsurface glow; the single-scatter albedo is `subsurface_color`.
A **hero-wavelength spectral-MIS estimator** samples one channel's mean-free-path per step and
reweights the others, and reduces exactly to the achromatic walk when the extinction is grey (clear
glass stays byte-stable). The walk runs on its own bounce budget (it does not starve surface
transport). Thin-walled subsurface keeps a diffuse reflection/transmission sheet.

**Transmission scattering** reuses the same volumetric walk: when `transmission_scatter` is set,
the smooth dielectric interior becomes a genuine scattering medium (milky/cloudy liquids) rather
than a fixed tint, while `transmission_color` / `transmission_depth` provide the Beer–Lambert
absorption.

### Color pipeline
- Scene-referred rendering in a selectable **working color space**: linear **Rec.2020**
  (default) or linear **Rec.709**, chosen per scene via `working_color_space` in the
  `[render]` table; assets (material colors, textures, environment maps) are converted
  automatically on load
- Physical camera exposure via **EV100** (`ev100` scene keyword)
- Physical environment scale via **`env_unit_nits`** (cd/m² per EXR unit)
- Tone mapping (shared Harmonia stage): switchable per scene — **AgX** (Troy Sobotka, wide DR, natural sun highlight rolloff), **ACES** RRT+ODT (Stephen Hill fit), **Reinhard** luminance, **Hable** / Uncharted-2 filmic
- Display output: SDR (sRGB), HDR10 (PQ/ST2084), scRGB — runtime negotiated with the swapchain
- Headless output: **EXR** is the scene-referred, untonemapped frame; **PNG** is the tonemapped version

### Bindless textures
- Descriptor set 1, binding 4: `COMBINED_IMAGE_SAMPLER` array (up to 1024 entries)
- `NonUniformResourceIndex` for correct divergent access
- Per-material texture maps, each converted to the render color space at load:
  `map_base_color`, `map_normal`, `map_orm` (packed occlusion/roughness/metalness,
  G=roughness/B=metalness), `map_emission_color`

### Scene format
TOML-based formats parsed by [Aether](https://github.com/McNopper/Aether): a
`<name>.scene.toml` scene description with companion `<name>.materials.toml` OpenPBR
material libraries (`model = "openpbr"`) and geometry-only OBJ meshes:

```toml
material_libraries = ["cornell.materials.toml"]

[render]
reference = "presets/preview.render.toml"      # shared preset; inline keys override
working_color_space = "lin_rec2020_scene"      # or "lin_rec709_scene"

[camera]
reference = "presets/cornell.camera.toml"      # translate / look_at / vfov / ev100

[tonemap]
tonemapper = "agx"                             # aces | agx | reinhard | hable

[[geometry]]
type = "instance"                              # instance | box | sphere
mesh = "cornell.obj"
materials = { Floor = "WhiteWall", LeftWall = "RedWall", RightWall = "GreenWall" }
```

OBJ files contribute **only geometry** — material import from OBJ/MTL is disabled by
design; all material assignments are declared in the scene file. See the
[Aether README](https://github.com/McNopper/Aether) for the full format reference.

---

## Architecture

Hyperion is the **offline / ground-truth** renderer in a family of four repositories:

| Repository | Role |
|------------|------|
| [Aether](https://github.com/McNopper/Aether) | GPU-agnostic file formats & scene data (`.scene.toml` / `.materials.toml` / OBJ → plain CPU structs); no Vulkan |
| [Harmonia](https://github.com/McNopper/Harmonia) | Shared Vulkan foundation reused **1:1** by both renderers — `harmonia::App` host, core/context, presentation, color management, tonemapping, bindless textures, shared GPU types, Slang shader build |
| **Hyperion** | This repo — offline path tracer (ground truth) |
| [Theia](https://github.com/McNopper/Theia) | Real-time forward renderer |

Hyperion consumes Aether and Harmonia via CMake `FetchContent`. The demo application is a
thin subclass of the shared **`harmonia::App`** host: Harmonia owns the window, swapchain,
HDR target, tonemapping/presentation, IBL probe and scene loading, while Hyperion injects
its renderer through the `harmonia::IRenderer` seam (Theia does the same). Slang shaders
are compiled at build time by Harmonia's shared `compile_slang_shaders` CMake rule
(`shaders/*.slang` → `build/shaders/*.spv`) and loaded through Harmonia's SPIR-V loader.
The **unidirectional path-integrator estimator** (emissive/env NEE + MIS + Russian roulette)
lives in Harmonia's shared `path_integrator.slang` and is reused **1:1** by both renderers
through an `ITracer` seam — Hyperion drives it with its ray-tracing-pipeline tracer, Theia
with inline `RayQuery`. The **GPU scene layout is renderer-specific**: Hyperion owns its own `Scene` and
`GpuInstance` (`src/hyperion/scene/`) built around index buffers and the ray-tracing
pipeline, distinct from Theia's meshlet layout. Only code shared 1:1 lives in Harmonia.

---

## Building

**Requirements:** Vulkan SDK 1.4, CMake 3.28+, Ninja, clang-cl, vcpkg.

```bash
cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-cl \
      -DCMAKE_CXX_COMPILER=clang-cl \
      -DCMAKE_TOOLCHAIN_FILE="<vcpkg-root>/scripts/buildsystems/vcpkg.cmake"

cmake --build build
```

---

## Running

```bash
# Headless render → EXR (scene-referred) + PNG (tonemapped)  (window is hidden)
build/hyperion.exe cornell_classic --output screenshots/cornell_classic.png

# Interactive window (default scene: cornell_classic)
build/hyperion.exe meadow_scene
```

### Command-line flags

| Flag | Default | Description |
|------|---------|-------------|
| `--scene <name>` / `-s` | `cornell_classic.scene.toml` | Scene name or path; bare names resolve against the assets directory (also accepted as first positional argument) |
| `--output <file>` / `-o` | — | Headless mode: accumulate and save EXR (untonemapped) + PNG (tonemapped), then exit; a `.png` output saves PNG only |
| `--spp <n>` | scene value | Override samples per pixel |
| `--depth <n>` | scene value | Override maximum bounce depth |
| `--width <n>` | 1920 | Override render width in pixels |
| `--height <n>` | 1080 | Override render height in pixels |
| `--validation` / `--no-validation` | disabled | Enable / disable Vulkan validation layers |

---

## Tests

Unit, component, module and integration suites:

```bash
cd build && ctest --output-on-failure
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| [Aether](https://github.com/McNopper/Aether) | Scene & material file formats (`.scene.toml` / `.materials.toml` / OBJ) — GPU-agnostic CPU data |
| [Harmonia](https://github.com/McNopper/Harmonia) | Shared Vulkan foundation (`harmonia::App` host, core, presentation, color, tonemapping, shared GPU types) |
| [Vulkan SDK](https://vulkan.lunarg.com/) | Ray tracing API |
| [volk](https://github.com/zeux/volk) | Vulkan loader |
| [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | GPU memory allocation |
| [SDL3](https://libsdl.org/) | Window & surface |
| [GLM](https://github.com/g-truc/glm) | Math |
| [OpenImageIO](https://openimageio.readthedocs.io/) | Image I/O — PNG/JPEG/EXR load and save (via Harmonia; stb and OpenEXR are transitive dependencies) |
| [Slang](https://shader-slang.com/) | Shader language |
| [Google Test](https://github.com/google/googletest) | Testing |

---

## References

The following specifications, textbooks, and learning resources informed the design of Hyperion:

### Rendering & Path Tracing
| Resource | Relevance |
|----------|-----------|
| [Physically Based Rendering: From Theory To Implementation, 4th ed.](https://www.pbrt.org/) (Pharr, Jakob, Humphreys) | Path tracing, BSDF sampling, MIS balance heuristic (§13.4.3), emissive area light NEE (§12.4), env map importance sampling via 2D separable CDF (§12.5) |
| [Veach — "Robust Monte Carlo Methods for Light Transport Simulation" (1997)](http://graphics.stanford.edu/papers/veach_thesis/) | Multiple Importance Sampling (MIS) — balance and power heuristics (§9.2); theoretical foundation for combining BSDF and NEE pdf estimates |
| [Shirley, Wang & Zimmerman — "Monte Carlo Techniques for Direct Lighting Calculations" (1996)](https://www.cs.utah.edu/~shirley/papers/tog96.pdf) | Uniform area sampling of triangles via sqrt-folding barycentric coordinates; area-to-solid-angle PDF conversion |
| [Henyey & Greenstein — "Diffuse Radiation in the Galaxy" (1941)](https://articles.adsabs.harvard.edu/pdf/1941ApJ....93...70H) | Henyey-Greenstein phase function for the subsurface / transmission volumetric random walk |
| [Wilkie, Nawaz, Droske, Weidlich & Hanika — "Hero Wavelength Spectral Sampling" (EGSR 2014)](https://cgg.mff.cuni.cz/~wilkie/Website/EGSR_14_files/WNDWH14.pdf) | Hero-wavelength spectral-MIS estimator for chromatic (per-channel) subsurface / transmission media |
| [Novák, Georgiev, Hanika & Jarosz — "Monte Carlo Methods for Volumetric Light Transport Simulation" (Eurographics STAR 2018)](https://cs.dartmouth.edu/~wjarosz/publications/novak18monte.html) | Free-flight distance sampling, collision estimators, and transmittance for the medium walk |
| [Ray Tracing Gems I & II](https://www.realtimerendering.com/raytracinggems/) (Haines et al., Marrs et al.) | Shadow ray precision, NEE techniques, ray tracing best practices |
| [Ray Tracing in One Weekend series](https://raytracing.github.io/) (Shirley et al.) | Introductory path-tracer architecture |

### Vulkan & Ray Tracing API
| Resource | Relevance |
|----------|-----------|
| [Vulkan Specification 1.4](https://registry.khronos.org/vulkan/specs/latest/html/) | `VK_KHR_ray_tracing_pipeline`, `VK_KHR_acceleration_structure`, descriptor indexing |
| [Khronos — Ray Tracing in Vulkan](https://www.khronos.org/blog/ray-tracing-in-vulkan) | Pipeline setup, SBT layout, shader stages |
| [NVIDIA — Ray Tracing Learning Library](https://developer.nvidia.com/rtx/ray-tracing) | Algorithm-level ray tracing techniques (implementation uses Khronos extensions only — no vendor-specific extensions) |
| [Slang Shading Language](https://shader-slang.com/) | `[raypayload]` semantic, `TraceRay`, Vulkan binding annotations |

### Material Model
| Resource | Relevance |
|----------|-----------|
| [OpenPBR Surface Specification v1.1.1](https://academysoftwarefoundation.github.io/OpenPBR/) | Material layer stack, parameter naming (base/specular/coat/fuzz/emission/transmission) |
| [MaterialX Standard Surface](https://materialx.org/) | Cross-reference for PBR parameter vocabulary |
| [Blender Principled BSDF](https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html) | Cross-reference for PBR parameter vocabulary |
| [Harmonia README — Surface BSDF references](https://github.com/McNopper/Harmonia#references) | Full citations for the shared BSDF closures (thin-film, sheen/LTC, MS-comp, conductor Fresnel) implemented in `bsdf_shared.slang` |

### Color Science
| Resource | Relevance |
|----------|-----------|
| [OpenColorIO](https://opencolorio.org/) | Color space transforms, ACES RRT/ODT, tonemapping nomenclature |
| [AgX by Troy Sobotka](https://github.com/sobotka/AgX) | AgX tone-mapping matrices and S-curve (MIT) |
| [ITU-R BT.2100](https://www.itu.int/rec/R-REC-BT.2100/) | PQ/ST2084 and HLG OETF for HDR display output |
| [IEC 61966-2-1 (sRGB)](https://www.color.org/srgb.xalter) | sRGB EOTF for SDR display output |

### Scene & Asset Formats
| Resource | Relevance |
|----------|-----------|
| [OpenUSD](https://openusd.org/release/api/index.html) | Naming conventions: Prim, Xform, Mesh, Material, Light, Camera, Instance |
| [glTF 2.0 Specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) | PBR material and scene graph conventions |
| [Wavefront OBJ](http://paulbourke.net/dataformats/obj/) | Geometry-only OBJ import (no MTL — materials are assigned in the scene TOML) |
