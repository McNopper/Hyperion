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
- Analytic spheres via `VK_KHR_ray_tracing_pipeline` intersection shaders
- Image-based lighting (IBL) — equirectangular HDR panorama via `env_map`
- Emissive mesh area lights (physical units: cd/m²)
- Firefly suppression (luminance clamping)
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
- Tone mapping: ACES RRT+ODT (Stephen Hill fit)
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

