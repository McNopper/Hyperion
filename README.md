# Hyperion

Vulkan path-tracer for OpenPBR.

> *[Hyperion](https://en.wikipedia.org/wiki/Hyperion_(mythology)) — Titan of heavenly light, father of Helios, Selene and Eos.*

Hyperion is a GPU path tracer built entirely on Vulkan 1.4 KHR ray tracing.  
It implements the [OpenPBR Surface v1.1](https://academysoftwarefoundation.github.io/OpenPBR/) material model and outputs a physically correct, linearly encoded HDR frame every render call.

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

| Thin-film | Meadow IBL | Textured Cube |
|:---------:|:----------:|:-------------:|
| ![openpbr_thinfilm](screenshots/openpbr_thinfilm.png) | ![meadow_scene](screenshots/meadow_scene.png) | ![textured_cube](screenshots/textured_cube.png) |

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

### Material model — OpenPBR Surface v1.1
All parameters follow the [OpenPBR spec](https://academysoftwarefoundation.github.io/OpenPBR/) naming:

| Layer | Parameters |
|-------|-----------|
| Base | `base_weight`, `base_color`, `base_roughness`, `base_metalness` |
| Specular | `specular_weight`, `specular_color`, `specular_ior`, `specular_roughness`, `specular_anisotropy` |
| Coat | `coat_weight`, `coat_color`, `coat_ior`, `coat_roughness`, `coat_darkening` |
| Fuzz | `fuzz_weight`, `fuzz_color`, `fuzz_roughness` |
| Emission | `emission_luminance`, `emission_color` |
| Thin-film | `thin_film_thickness`, `thin_film_ior` |
| Transmission | `transmission_weight` |
| Geometry | `geometry_opacity` |

### Color pipeline
- All internal calculations in **linear Rec.2020**
- Physical camera exposure via **EV100** (`ev100` scene keyword)
- Physical environment scale via **`env_unit_nits`** (cd/m² per EXR unit)
- Tone mapping: switchable per scene — **AgX** (Troy Sobotka, wide DR, natural sun highlight rolloff), **ACES** RRT+ODT (Stephen Hill fit), **Reinhard** luminance, **Hable** / Uncharted-2 filmic
- Display output: SDR (sRGB), HDR10 (PQ/ST2084), scRGB — runtime negotiated with the swapchain

### Bindless textures
- Descriptor set 1, binding 4: `COMBINED_IMAGE_SAMPLER` array (up to 1024 entries)
- `NonUniformResourceIndex` for correct divergent access
- Base-color texture support (`map_base_color` in MTL)

### Scene format
Line-based text format (`.scene`) inspired by Wavefront OBJ/MTL:

```
mtllib cornell.mtl          # load material library

camera
  translate  278  273  -800
  look_at    278  273   279
  vfov       39.1

ev100        7.0            # physical camera exposure
spp          64             # samples per pixel
max_depth    8

o cornell.obj               # load geometry (pure geometry — no materials inside OBJ)
  material Floor     WhiteWall   # assign material per OBJ group
  material LeftWall  RedWall
  material RightWall GreenWall

sphere  60.0
  usemtl Glass
  translate  430  60  200

env_map       meadow_2_4k.exr
env_unit_nits 10000
tonemapper    agx             # aces (default) | agx | reinhard | hable
```

OBJ files contain **only geometry** — all material assignments are declared in the scene file.

---

## Building

**Requirements:** Vulkan SDK 1.4, CMake 3.25+, Ninja, clang-cl, vcpkg.

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
# Headless render → PNG
build/hyperion.exe --scene assets/cornell_classic.scene \
                   --output screenshots/cornell_classic.png \
                   --headless

# Interactive window
build/hyperion.exe --scene assets/meadow_scene.scene
```

---

## Tests

107 tests across unit, component, module and integration suites:

```bash
cd build && ctest --output-on-failure
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| [Vulkan SDK](https://vulkan.lunarg.com/) | Ray tracing API |
| [volk](https://github.com/zeux/volk) | Vulkan loader |
| [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | GPU memory allocation |
| [SDL3](https://libsdl.org/) | Window & surface |
| [GLM](https://github.com/g-truc/glm) | Math |
| [stb_image](https://github.com/nothings/stb) | PNG/JPEG load |
| [tinyexr](https://github.com/syoyo/tinyexr) | EXR load/save |
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
| [OpenPBR Surface Specification v1.1](https://academysoftwarefoundation.github.io/OpenPBR/) | Material layer stack, parameter naming (base/specular/coat/fuzz/emission/transmission) |
| [MaterialX Standard Surface](https://materialx.org/) | Cross-reference for PBR parameter vocabulary |
| [Blender Principled BSDF](https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html) | Cross-reference for PBR parameter vocabulary |

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
| [Wavefront OBJ](http://paulbourke.net/dataformats/obj/) | Geometry-only OBJ import (no MTL — materials are assigned in the `.scene` file) |
