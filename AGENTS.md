# AGENTS.md — Hyperion

Short orientation for coding tools that read a repository description automatically. The
complete documentation is in [README.md](README.md); each section below points into it.
Outstanding work and the release history are in [PLAN.md](PLAN.md).

## What this repo is

Hyperion is the offline path tracer and the reference renderer of a five-repository family
(slang-math → Aether → Harmonia → Hyperion and Theia). Its images are the ground truth that Theia
is aligned to. Summary: [README.md, Architecture](README.md#architecture).

- Code that both renderers use unchanged lives in Harmonia: the OpenPBR BSDF
  (`Harmonia/shaders/bsdf_shared.slang`), the surface estimator
  (`Harmonia/shaders/path_integrator.slang`), the medium-walk mathematics
  (`Harmonia/shaders/medium_walk.slang`) and the application host `harmonia::app::App`.
- Hyperion contributes the ray-tracing shaders in `shaders/`, its GPU scene layout in
  `src/hyperion/scene/`, the per-frame recording in `src/hyperion/renderer/` and the host
  subclass in `src/demo/`. Its classes live in `namespace hyperion`.
- Hyperion never depends on Theia. Code that both renderers need moves down into Harmonia.
- The OpenPBR conformance tests of the shared BSDF are `tests/unit/test_bsdf.cpp`.

## Running

```powershell
build/hyperion.exe --scene cornell_classic --output out.exr           # headless: EXR + PNG
build/hyperion.exe --scene dragon_teapot --spp 256 --output ref.exr   # clean reference
build/hyperion.exe shaderball_base                                     # interactive window
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--spp <n>` | the scene's `samples_per_pixel`, else 4 | Samples per pixel. A capture accumulates N single-sample frames. |
| `--depth <n>` | the scene's `max_depth`, else 8 | Maximum number of surface interactions per path. |
| `--sms-mode <off\|local>` | `local` | `off` renders with ordinary path tracing only, without specular manifold sampling (the reference mode for manifold-sampling comparisons). |

Every other flag is parsed by Harmonia; the list is in
[README.md, Command-line flags](README.md#command-line-flags). There is no `--offscreen` flag:
`--output` selects headless mode. Shaders are loaded from the shader directory compiled into the
executable (`HYPERION_SHADER_DIR`), never from the working directory.

## Gotchas

- Assets are read from the Aether checkout that CMake resolved: `build/_deps/aether-src/` by
  default, otherwise the directory given to `FETCHCONTENT_SOURCE_DIR_AETHER`. An edit in another
  Aether working tree does not change a render.
- A scene needs a camera placement (translation and rotation, inline or through a preset). The
  scene loader rejects a scene without one and has no default view.
- OBJ faces must be wound outward. Emissive-triangle normals derive from the winding, so an
  inward-wound emitter renders black.
- Reference renders use 256 samples per pixel and are compared as EXR, never as PNG. A
  low-sample reference measures its own noise. Details:
  [README.md, Producing a reference render](README.md#producing-a-reference-render).
- A capture of an animated scene needs `--sequence-frames`. Without it the accumulation restarts
  at every sample and the image is a single-sample snapshot of the last animation time.
- A scene with a `[haze]` section renders with its distance haze: the ray-generation shader
  composites each sample through the shared `Harmonia/shaders/haze.slang`, and the ray payload
  carries the distance to the primary surface for it (`primaryT` in `shaders/common.slang`).
  Shadow and secondary rays do not pass through the medium. Theia applies the same module, so
  the haze is compared like every estimator change.
- With `--glare` the PNG of a capture carries Harmonia's lens glare and the EXR does not. A
  reference render is an EXR and is never affected by it.

## Test scenes

Cheap for iteration: the Cornell scenes (`cornell_classic` is the default scene). Material
fixtures: `openpbr_*` and `shaderball_*`. Do not use `ABeautifulGame` for iteration; it is
reserved for gallery renders. Table: [README.md, Test scenes](README.md#test-scenes).

## Build & test

```powershell
cmake --preset win              # or the explicit configure in README.md, Building
cmake --build --preset win
ctest --preset win              # or: cd build; ctest --output-on-failure
```

- `ctest -LE analysis` runs the test suites only. `ctest -L analysis` runs the source checks:
  clang-tidy, the architecture contract and the dead-code check.
- All tests are serialized, and GPU jobs never overlap: one render or one test suite at a time.
- Before a push: a warning-clean build, the test suites, a representative render without
  validation messages, and for a lint-affecting change the lint commands of the continuous
  integration.

## Conventions

- The estimator is unbiased: no clamp, special case or scene-specific behavior in the render
  path.
- The shared accumulation pass is the only owner of the frame history. The ray-generation
  shader writes the current dispatch only.
- Defects are fixed in the code that owns them. A tolerance is never loosened to pass a test.
- GPU-driven, latest standard Vulkan, cross-vendor only: core and `KHR`/`EXT` extensions, no
  vendor extension, no fallback path. The working color space is scene-referred and linear.
- Commits stay in the repository they belong to. Nothing is pushed unless requested.

## Rules when changing the renderer

- `PathTracer::render()` records bounded work. Before an image is read, accumulated or exported,
  the caller submits it, waits, and repeats `frameReady()` and `recordContinuation()` until the
  frame is ready. The host does this through its `IFrameCompletion` interface; a failed check
  fails the frame instead of accumulating a partial result.
- The ray dispatch is indirect (`vkCmdTraceRaysIndirect2KHR`); `VK_KHR_ray_tracing_maintenance1`
  is a hard device requirement. The indirect command buffer is written at creation and on resize
  only. Acceleration structures are built with device-side commands only.
- The ray-tracing pipeline is created by Harmonia (`Harmonia/src/harmonia/renderer/Pipeline.cpp`)
  with four hit groups in a fixed order: triangle radiance, triangle shadow, sphere radiance,
  sphere shadow. Only the shadow hit groups carry an any-hit shader.
- The manifold-sampling sampler is shared: `shaders/sms_sample.slang` includes
  `Harmonia/shaders/sms_compute.slang`, and the path classifier is
  `Harmonia/shaders/sms_path.slang`. Hyperion holds only the payload and scene adapters; a second
  sampler must not be added. The first receiver after a specular prefix, the relative depth and
  the pure-absorber exit are classification rules that new consumers must preserve.
- `ARCHITECTURE.md` is machine-checked. Its `arch` block and the diagram section between the
  `check_arch` markers stay unchanged; only the prose around them is edited. Details:
  [ARCHITECTURE.md](ARCHITECTURE.md).
