# Tools

The scripts in this directory check and support the repository. The Python scripts need
Python 3 and read the repository only, except where the table says otherwise. How the checks
fit into the test workflow is described in [README.md, Tests](../README.md#tests).

| Script | Purpose | Writes |
|--------|---------|--------|
| `check_arch.py` | Checks the include graph of `src/` against the contract in [ARCHITECTURE.md](../ARCHITECTURE.md) | Nothing, except with `--diagram` or `--derive --out` |
| `check_dead.py` | Finds dead code: orphan headers, never-compiled sources, unused `FetchContent` declarations | Nothing |
| `check_tidy.py` | Runs clang-tidy in parallel over the translation units and classifies the findings | A log file next to the compile database |
| `coverage.py` | Builds an instrumented tree, runs the test suites and reports the source coverage of `src/` | The coverage build directory |
| `generate_gallery.ps1` | Renders the 30 screenshot scenes of the README gallery | The screenshot directory |
| `check_exr.cpp` | Source of a small OpenEXR sanity inspector; not built by CMake | Nothing |

`check_dead.exceptions` lists the justified findings of the dead-code check.

---

## check_arch.py

The architecture check. The `arch` block of [ARCHITECTURE.md](../ARCHITECTURE.md) declares
layers, modules and the permitted includes; the script extracts the include graph of `src/` and
compares the two. What the contract means and how to change it is described in that document.

```powershell
python tools/check_arch.py --repo . --doc ARCHITECTURE.md
```

| Argument | Effect |
|----------|--------|
| `--repo DIR` | Repository to check. Default: the directory that contains `tools/`. |
| `--doc FILE` | Contract document. Default: the first existing `ARCHITECTURE.md` in the repository's `docs` directory, in the repository root or next to this script. |
| `--report` | Also prints the layer stack, the declared edges, every measured module edge with its include count, the external include roots and the refactoring smell proxies. |
| `--json` | Prints the same summary as JSON instead. |
| `--diagram` | Regenerates the diagram section of the document from the rules. This is the only mode that rewrites `ARCHITECTURE.md`. |
| `--derive` | Writes a draft contract that describes the code as it is, to standard output or to the file given with `--out FILE`. An existing draft is never overwritten when nothing was derived. `--scan DIRS` selects the scanned roots. |
| `--selftest` | Runs the script's own test on synthetic repositories in a temporary directory. |
| `--version` | Prints the engine version. |

Output:

| Line | Meaning |
|------|---------|
| `architecture clean (…)` | The include graph matches the contract. |
| `must not include … (edge not declared)` | An include that no `allow` line permits. |
| `must not depend on … - the layer stack flows one way` | A declared edge that runs against the layer order. |
| `unclassified module: …` | A scanned file in no declared module. |
| `must not include … (forbidden dependency)` | An include of a forbidden path. |
| `include cycle: …` | Files that include each other directly or indirectly. |
| `DIAGRAM DRIFT` | The diagram section no longer matches the rules; run `--diagram`. |
| `HEADER SIZE: …` | A file that reached the line limit. |

Exit codes: `0` clean, `1` violations, `2` the contract or the invocation has an error (an
unknown or duplicate knob, an edge to an undeclared module, a missing `arch` block), `77`
nothing to scan.

ctest registers the same command as `test_arch` (label `analysis`) where the check-arch engine
is installed in the user profile.

## check_dead.py

The dead-code check. It looks for three kinds of findings:

| Check | Finding |
|-------|---------|
| `orphan-headers` | A header that no other scanned file includes. |
| `never-compiled` | A source file that is neither in the newest compile database below the repository nor mentioned by name in a CMake file. |
| `fetchcontent` | A `FetchContent_Declare` whose name no `FetchContent_MakeAvailable` or `FetchContent_Populate` uses. |

```powershell
python tools/check_dead.py --family ..
```

| Argument | Effect |
|----------|--------|
| `--repo DIR` | Repository to check. Default: the directory that contains `tools/`. |
| `--family DIR` | Directory with the sibling repositories. A header that nobody includes in this repository is reported as consumed when a sibling includes it. ctest passes the parent directory of the repository. |
| `--report` | Also prints the scanned roots, the file counts and the compile database in use. |
| `--version` | Prints the engine version. |

Output: one `PASS` or `FAIL` line per check with the number of findings, followed by one line
per finding, and a final `check_dead: clean` or `check_dead: <n> finding(s)`. Rescued headers
appear as `info cross-repo consumer: …`.

Exit codes: `0` clean, `1` findings, `77` nothing to scan.

A finding needs a decision before anything is deleted: real dead code is removed; a justified
case gets a line in `tools/check_dead.exceptions`:

```text
<check> <path or name> <reason>
```

## check_tidy.py

Runs clang-tidy over every project translation unit listed in the compile database, in
parallel, and classifies the findings by the categories configured in `.clang-tidy`: the checks
`clang-diagnostic-*`, `clang-analyzer-*` and `bugprone-*` fail the run; `modernize-*`,
`performance-*` and `portability-*` are listed only. Translation units below `build/`,
`_deps/` and third-party directories are skipped.

```powershell
python tools/check_tidy.py [--db PATH] [--jobs N]
```

| Argument | Effect |
|----------|--------|
| `--db PATH` | Compile database. Default: `build/compile_commands.json`. |
| `--jobs N` | Parallel clang-tidy processes. Default: CPU count minus 2, at least 4. |

clang-tidy is resolved from the newest Visual Studio installation (through `vswhere`), with a
fixed fallback path for the Visual Studio LLVM component; the script passes the clang-cl
driver mode. If neither location holds `clang-tidy`, it exits with `77`.

Output: a progress line every ten translation units, the list of report-only findings (the ten
most frequent checks), and then either `clang-tidy clean (…)` or `ERROR-CATEGORY FINDINGS (n)`
with one line per finding and the path of the full log (`clang_tidy.log` next to the compile
database).

Exit codes: `0` clean, `1` findings of an error category, `77` clang-tidy or the compile
database is missing (ctest then reports `test_tidy` as skipped).

## coverage.py

Builds an instrumented copy of the project, runs the test suites with profiling enabled, and
prints the coverage report of `src/`. It needs a Clang toolchain with `llvm-profdata` and
`llvm-cov`; the tools are looked up next to the coverage build's compiler first and then on
the search path.

```powershell
python tools/coverage.py
```

| Argument | Effect |
|----------|--------|
| `--build DIR` | Coverage build directory. Default: `build/coverage`. |
| `--skip-build` | Skip configure and build; only run the suites and the report. |
| `--show` | Print annotated sources (`llvm-cov show`) instead of the summary (`llvm-cov report`). |

The script configures the coverage build with `-DHYPERION_COVERAGE=ON` and, when a main build
directory `build/` exists, copies its generator, compiler and toolchain settings so that both
builds use the same dependencies. It then builds, runs `ctest -LE analysis` with
`LLVM_PROFILE_FILE` set, merges the profiles and prints the report for every test executable
that ctest lists.

Exit codes: `0` report produced, `1` a suite or the coverage run failed, `77` `llvm-profdata` or
`llvm-cov` not found, `2` argument error.

The test suites need a GPU; coverage is therefore a local activity and not part of CI.

## generate_gallery.ps1

Renders the 30 scenes of the README gallery, one after the other.

```powershell
.\tools\generate_gallery.ps1 [-Renderer path\to\hyperion.exe] [-OutputDir screenshots] [-Spp 64] [-Width 1280] [-Height 720]
```

Each scene is rendered headless with `--scene <name> --spp <n> --width <w> --height <h> --output <OutputDir>\<name>.png`; a `.png` output path writes the PNG only. The script prints `[i/total] <name> ...` for every scene with the first two lines of the renderer's output that contain `Saved` or `ERROR`, warns when a scene exits with a non-zero code, and ends with `Gallery complete: 30 scenes -> <OutputDir>`.

The tracked gallery is compared byte for byte: when nothing in the rendering path changed, a regeneration reproduces the tracked PNGs exactly (compare with `git status`). The full gallery saturates the GPU; no other GPU job runs at the same time.

## check_exr.cpp

The source of a small OpenEXR inspector, compiled by hand when needed and not part of the CMake build (it is listed in `tools/check_dead.exceptions` for this reason). It needs the OpenEXR and Imath headers.

```text
check_exr <file.exr>
```

It prints the image size and the number of NaN, infinite and negative values, and the largest finite value. The default argument is `Aether/assets/meadow_2_4k.exr`.
