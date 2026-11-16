# Hyperion v0.9.0 (2026-11-16) - minor release

Offline path tracer (ground truth) release consuming **Harmonia v0.9.0 + Aether v0.9.0**.

## Shipped
- **Deformation (ANI4/ANI5 renderer half)**: the shared `DeformPass` runs from the animation
  tick (stage -> rest->morph->skin dispatch -> mesh-local copy -> BLAS refit), with the
  `scene.vertexRest` rest copy and `meshSceneVertexOffsets` substrate. **DQS** rides the shared
  job/staging with zero renderer code. The ground-truth renderer keeps bit-exact estimator
  properties (no estimator change; deformation only rewrites skinned/morphed vertices).
- **Parity gate**: recalibrated thresholds (mean<=120 / rel<=250 / SSIM>=0.6 / hist>=0.8) -
  fresh manifest run **14/14 PASS** (this renderer is the reference side; 0 VUID).
- Triage sync: C11 consumed-claim removed, C12 confirmed shipped.

## Receipts
- Unit suite **43/43**; end-to-end sequence renders (skin/morph/DQS fixtures): exit 0,
  **0 VUID / 0 validation messages**, motion proof (frame0 vs frame2 mean_diff 10.1-10.3).

## Release mechanics
- Local commits only: **tag `v0.9.0`, push commits + tag together** (pins reference the new
  Harmonia/Aether tags; local builds use `FETCHCONTENT_SOURCE_DIR_*` overrides).

## Also in this release (added after the notes assembly)

- **ANI8 half** (commit c70dbf7): per-tick camera-from-node placement in
  `record()` (setPosition/setTarget/setUp/setNearFar, own tick guard) and the
  offline movie-image path `--sequence-frames 1 --frame N` = one fully
  accumulated image at tick N. Receipts: 43/43, `--frame 0/58` pair mean_diff
  122.9 (1.7% from Theia's 124.9), 0 VUID on every run.
- **One animation clock + time-based capture (2026-11-16):** `--capture-every-ms`
  / `--start-seconds` (with `--sequence-frames`) sample animation time at
  fixed intervals for movie-frame ranges; scene/deform/TLAS/camera share
  `App::animationClock`. Receipts: 43/43, interval == tick-grid equivalence
  0.000, 0 VUID.

