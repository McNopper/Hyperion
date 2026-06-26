# AGENTS.md — Hyperion

Quick-start context for AI agents so basic facts don't have to be rediscovered each session.

## What this repo is

**Hyperion** is a Vulkan **path tracer** and the **ground-truth reference renderer** for the
pipeline. It uses ray tracing (BLAS/TLAS), index buffers, NEE + MIS with environment
importance sampling (CDF). Other renderers (Theia) are aligned to match Hyperion.

**Material model = OpenPBR Surface** (Academy Software Foundation), whose canonical/reference
implementation is **MaterialX** (`mx_*` genGLSL nodes). Hyperion's BSDF (the shared Harmonia
`bsdf_shared.slang`) is the **conformance ground truth** for OpenPBR in this pipeline; when
improving spec-correctness, fix it here first, regenerate references, then align Theia. It is
structurally faithful but not yet 100% complete (e.g. sheen, GGX multi-scatter compensation,
true volumetric SSS are approximations) — cross-check parameters/behaviour against MaterialX.

Pipeline (dependency direction):

```
Aether (file format)  ->  Harmonia (shared Vulkan lib)  ->  Hyperion (this repo, ground truth)
                                                         \-> Theia    (real-time rasterizer)
```

Consumes Aether + Harmonia via CMake FetchContent. The demo is a thin `harmonia::App`
subclass injecting `harmonia::IRenderer`.

## Running

```powershell
build/hyperion.exe --scene cornell_classic --output out.exr            # headless EXR+PNG
build/hyperion.exe --scene fixture_ibl --spp 512 --output ref.exr      # clean IBL reference
build/hyperion.exe meadow_scene                                        # interactive window
```

CLI flags: all common Harmonia flags (`--scene/-s`, `--output/-o`, `--width`, `--height`,
`--validation`/`--no-validation`) **plus** Hyperion-only:

| Flag | Default | Meaning |
|------|---------|---------|
| `--spp <n>` | scene preset value | Override samples per pixel |
| `--depth <n>` | scene preset value | Override max bounce depth |

⚠️ No `--offscreen` flag — headless is triggered by `--output`.

## Gotchas (these waste a cycle every time they're forgotten)

- **Assets come from `build/_deps/aether-src/assets/`** (FetchContent clone), NOT the working
  Aether tree. Editing `C:\Development\GitHub\Aether\assets` does nothing unless you also
  update the `_deps` copy or build with `-DFETCHCONTENT_SOURCE_DIR_AETHER=...`. Symptom:
  two "different" renders give byte-identical metrics. See Aether/AGENTS.md.
- **spp for parity:** scenes using `alignment_16spp_8bounce.render.toml` render at only
  16 spp (noisy under IBL). Pass `--spp 512` when producing a parity reference, or the diff
  measures Monte-Carlo noise, not a real discrepancy.
- **Emissive winding:** emissive-triangle normals derive from OBJ winding
  (`cross(edge1,edge2)`); back-facing emitters are skipped in NEE. OBJs must be
  outward-facing (CCW-from-outside) or they render black.

## Test scenes

- Quick parity/iteration (cheap): `cornell_classic`, `cornell_spheres`, `cornell_suzanne`,
  `dragon_teapot`.
- **Never** use `ABeautifulGame` for quick test renders — it is expensive.
  (It is required only in final screenshot/render *deliverable* batches, not iteration.)

## Build & test

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl `
      -DCMAKE_TOOLCHAIN_FILE="<vcpkg-root>/scripts/buildsystems/vcpkg.cmake"
cmake --build build
cd build; ctest --output-on-failure
```

SDL3, slangc and volk come from the Vulkan SDK (not vcpkg). vcpkg provides openexr, stb, glm.

## Conventions

- Commit, but do **not** push unless asked.
- **GPU-driven, latest standard Vulkan, cross-vendor only** (core + `KHR`/`EXT`). No
  vendor-specific extensions (`VK_NV_*`/`VK_AMD_*`/`VK_INTEL_*`) — must run on any vendor.
- Working color space is scene-referred (e.g. `lin_rec2020_scene`).
