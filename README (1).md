# GPU-Based Connected-Component Labeling with CUDA

8-connected image labeling with GPU-side convergence detection, parallel component statistics, an independent CPU reference, and a measured Tesla T4 benchmark sweep.

`CUDA C/C++` · `Python` · `Connectivity` · `Algorithms` · `Validation` · `Benchmark`

**IIT Jodhpur · PG Diploma in AI · GPU Programming · Problem 3**
Team: Ritesh Maury (G25AIT2087) · Shrikrishna Tripathi (G25AIT2103) · Alok Sahni (G25AIT2138)

---

## Submission links

| Item | Link | Status |
|---|---|---|
| GitHub repository | https://github.com/riteshmaury-iitj/gpu-assignment | Live — public |
| Colab notebook (live CUDA lab, T4) | https://colab.research.google.com/drive/19ZL9M_L7r8LkQH4GNw7onE1wsR_mhVQL | Live — runs end to end on a T4 |
The four "not applicable" rows have no counterpart in this source tree. If the submission template requires them, the artefacts have to be produced first (a Kaggle notebook that compiles and runs the CUDA lab on a T4, a Docker image wrapping the `nvcc` build and the verification matrix, W&B logging of the sweep in `python/benchmark.py`). Do not link a bare host domain in place of a missing artefact.

The report, handbook, 18-slide presentation, narrated video, Colab notebook, browser demo, results CSVs and guides travel inside the submission archive (`01_Report` … `09_Guides`) and need no external link.

---

## Overview

Connected-Component Labeling (CCL) groups adjacent foreground pixels in a binary image into distinct objects. This project implements an 8-connected CCL pipeline in CUDA C/C++, with an independent single-threaded CPU baseline, a serial twin that compiles the same per-pixel source as the kernels, and Python tools for image generation, verification, visualization and benchmarking.

For each foreground component the output is a compact integer label, area, centroid and axis-aligned bounding box. This is an educational implementation, not a production medical-imaging or vision system.

### Two revisions, two evidence sets — read this before quoting a number

This repository has moved ahead of the revision that produced the published benchmark, and the two must not be conflated:

| | **Current source tree** (`src/`) | **Colab lab** (legacy single-file) |
|---|---|---|
| Implementation | `ccl_core.h` + `ccl_kernels.cuh` + `ccl_cuda.cu`, 14 kernels, three selectable algorithms | one self-contained `ccl_cuda.cu`, 6 kernels |
| Thread mapping | 2-D tile 32×8, 1-D kernels at 256 | 2-D block 16×16, grid 64×64 at 1024² |
| Label updates | every write an `atomicMin`, every concurrent read `ccl_load_relaxed()` | in-place non-atomic reads/writes |
| Compaction | deterministic parallel exclusive prefix sum | `1 + atomicAdd(d_count)` on a device counter |
| Convergence flag | device `atomicMax` | device `atomicExch` |
| Validation | byte-exact `memcmp` against CPU BFS, serial twin and all algorithms | partition-equivalence vs CPU BFS and SciPy |
| Correctness evidence | 480/480 CPU + serial-twin checks; 636/636 device-inclusive and 20/20 repeat-determinism on a **previous** revision | PASS on blobs, shapes and 3 of 4 edge cases on a T4 |
| Timing evidence | **none on the current revision** | full T4 sweep, reproduced 2026-10-28 (below) |

**Every timing number in this README comes from the Colab lab.** The current source tree has not been re-run on a GPU, and two `initcheck` findings from an earlier T4 run are unresolved. No speed-up is claimed for the current revision until it is re-measured.

---

## Key features

- **8-connectivity** — horizontal, vertical and diagonal foreground neighbours are connected.
- **Three independent strategies** — `--algo prop` (in-place iterative propagation, the default), `--algo dbuf` (double-buffered, race-free by construction), `--algo uf` (non-iterative union-find). All three must produce byte-identical output.
- **GPU-resident convergence flag** — a device atomic flag; 4 bytes per iteration cross the bus instead of the full label image (64 MB at 4096², a factor of ~16 million).
- **Atomic-only label updates** — every write is an `atomicMin`, every concurrent read goes through `ccl_load_relaxed()`; `make cuda-strict` substitutes libcu++ `cuda::atomic_ref` for exactness against the CUDA memory model.
- **Deterministic compaction** — labels are compacted to `1..K` by a hand-written recursive parallel exclusive prefix sum, not an atomic counter, so component ids are a pure function of the input.
- **Parallel component statistics** — `atomicAdd` (64-bit coordinate sums), `atomicMin` and `atomicMax` for area, centroids and inclusive bounding boxes; results are bit-reproducible.
- **Byte-exact validation** — because the fixed point is unique, CPU BFS, the serial twin and all three GPU algorithms are compared with `memcmp`, not partition equivalence.
- **GPU-free verification** — NVRTC device compilation, clang host-only type-checking against a declarations-only CUDA shim, and a PTX audit confirming the intended atomics were emitted.

---

## Technology stack

| Layer | Tools / techniques |
|---|---|
| GPU implementation | CUDA C/C++, CUDA Runtime API, kernel launches, device atomics, pinned host memory, streams and events |
| CPU reference | C99, BFS flood fill, serial label propagation |
| Serial twin | C++11, compiles the same per-pixel bodies as the kernels, permutable visit order |
| Test and analysis | Python 3, NumPy, SciPy (`ndimage`), Matplotlib, pandas |
| Static verification | NVRTC, clang (CUDA host-only mode), PTX audit, `compute-sanitizer` |
| Development | Git/GitHub, GNU Make, GCC, NVIDIA `nvcc`; Windows build script |
| Target GPU | NVIDIA Tesla T4 (`sm_75`); other CUDA devices with a compatible `-arch` |

---

## Architecture and processing workflow

**Connectivity rule.** For pixel `(x, y)`, consider offsets `dx, dy ∈ {-1, 0, 1}` excluding `(0, 0)`, keeping only in-bounds foreground neighbours.

**Statistics for component k.** Area is its pixel count; `cx = sum(x) / area`, `cy = sum(y) / area`; the bounding box is `(xmin, ymin, xmax, ymax)` in inclusive pixel coordinates.

**Pipeline.** Upload → initialise labels → propagate (loop, with periodic pointer-jumping) → compact ids to `1..K` → component statistics → download. Everything between the upload and the final download stays on the device; only the 4-byte convergence flag crosses the bus per iteration.

### Why the in-place version is race-safe

Two invariants carry the argument. **(I1)** `labels[L-1] ≤ L` for every foreground label — updates go only through `atomicMin`, so a label can only decrease, and only to a value already held by a pixel in the same component. **(I2)** following `parent(L) = labels[L-1]` strictly decreases until a root is reached, so the chain is acyclic and `find_root` always terminates even while other threads mutate the array.

A stale read is therefore harmless: a reader can observe only a value ≥ the current one, and any value it observes was at some point a legitimate label of a pixel in that same component — so a stale read can delay convergence but cannot merge two components or split one.

This is an argument, not a proof that the code implements it. `--cross-check`, the 20× repeat-determinism test and `make sanitize` are the hardware checks that test it, and the two unresolved `initcheck` findings are listed under [Known issues](#known-issues).

---

## Quick start

### Prerequisites

- Python 3 with `numpy`; `scipy` for the reference, `matplotlib` for plots, `pandas` for benchmarks.
- GCC or another supported C compiler for the CPU executable and the serial twin.
- To run the CUDA executable: an NVIDIA CUDA-capable GPU, a supported driver, and CUDA Toolkit with `nvcc` on `PATH`.

```bash
# Clone
git clone https://github.com/riteshmaury-iitj/gpu-assignment.git
cd gpu-assignment

# Python analysis dependencies
python3 -m pip install numpy scipy matplotlib pandas

# CPU reference + serial twin (no CUDA needed)
make

# CUDA target, on a machine with the toolkit
make cuda ARCH=sm_75
```

On Windows, open an **x64 Native Tools Command Prompt for Visual Studio** with `nvcc` available and run `build_windows.bat sm_75`. This path has not been independently re-tested for the current revision.

### Example: generate, label, verify, visualize

```bash
# 1. Deterministic 512x512 binary image
python3 python/gen_image.py blobs.bin 512 512 --kind blobs --density 0.3 --seed 42 --png blobs.png

# 2. CPU reference
./ccl_cpu --image blobs.bin --algo bfs --out-labels labels_cpu.bin --out-stats stats_cpu.csv

# 3. CUDA implementation (NVIDIA GPU required)
./ccl_cuda --image blobs.bin --out-labels labels_gpu.bin --out-stats stats_gpu.csv --verbose

# 4. Compare against the CPU reference — byte-exact, not just partition-equivalent
python3 python/verify.py blobs.bin labels_gpu.bin stats_gpu.csv --ref-labels labels_cpu.bin

# 5. Visualization
python3 python/visualize.py blobs.bin labels_gpu.bin stats_gpu.csv components.png
```

CPU-only verification: run step 2, then `python3 python/verify.py blobs.bin labels_cpu.bin stats_cpu.csv`.

### Verification targets

```bash
make test        # verification matrix, CPU + serial twin        → 480 / 480
make test-cuda   # the same matrix including the GPU             → needs a device
make check       # NVRTC device compile + clang host type-check  → no GPU required
make sanitize    # compute-sanitizer racecheck/memcheck/synccheck/initcheck
```

**Output formats.** Images and labels use an 8-byte header (`int32` width, `int32` height) followed by row-major pixels (`uint8`) or labels (`int32`). CSV columns are `id,area,cx,cy,xmin,ymin,xmax,ymax`. Centroids are written with `%.12g` — at `%.6f` a value such as `198.118055555…` loses enough precision to fail an exactness check as a pure artefact of the text format.

---

## Benchmark

```bash
mkdir -p results figures
python3 python/benchmark.py --exe ./ccl_cuda --cpu ./ccl_cpu \
  --sizes 256,512,1024,2048,4096 --kinds blobs,noise,shapes --reps 5 \
  --out results/benchmark.csv
python3 python/plot_results.py results/benchmark.csv figures/fig5_gpu_benchmark.png
```

**Timing scopes are not interchangeable.** The programs report `label_ms` (CUDA events around the propagation loop), `label_plus_compact_ms`, `stats_ms` and `end_to_end_ms` (host wall clock including allocation, H2D, all kernels, D2H and file writes). Only `end_to_end_ms` is comparable with the CPU program's `end_to_end_ms`. `benchmark.py` emits `speedup_label` and `speedup_end_to_end` separately and never mixes scopes — any speed-up quoted must name its scope.

`--cross-check` triples the work and must be off when benchmarking.

---

## Measured results — Tesla T4

**Run environment.** Google Colab, Tesla T4 (`sm_75`, 40 SMs, 15 360 MiB), driver 580.82.07, CUDA 13.0, compiled `nvcc -O3 -arch=sm_75` and `gcc -O3`. Sweep executed 2026-10-28. Three repetitions per point; minimum time reported, iteration count the maximum across repetitions.

**Scope of these numbers.** The GPU figure is the CUDA-event span from `init_labels` through `component_stats` — it excludes the host-to-device upload, the result download and all file writes. The CPU figure is host wall clock over labelling plus statistics, also excluding file I/O. The two are comparable to that extent and no further; neither is a true end-to-end number, and neither includes PCIe transfer. **These were produced by the legacy single-file Colab revision, not by the current `src/` tree.**

### blobs — compact discs, 30 % foreground

| Size | Components | Iterations | CPU BFS (ms) | GPU compress (ms) | GPU no-compress (ms) | Speed-up |
|---|---|---|---|---|---|---|
| 256² | 143 | 32 | 1.297 | 1.145 | 2.349 | 1.1× |
| 512² | 223 | 69 | 4.738 | 2.643 | 5.795 | 1.8× |
| 1024² | 250 | 49 | 17.795 | 4.158 | 13.123 | **4.3×** |
| 2048² | 264 | 85 | 66.655 | 29.912 | 81.547 | 2.2× |
| 4096² | 276 | 161 | 315.356 | 129.479 | 528.891 | 2.4× |

### noise — i.i.d. pixels, 50 % foreground

| Size | Components | Iterations | CPU BFS (ms) | GPU compress (ms) | GPU no-compress (ms) | Speed-up |
|---|---|---|---|---|---|---|
| 256² | 244 | 29 | 3.713 | 1.198 | 11.432 | 3.1× |
| 512² | 876 | 33 | 14.750 | 2.032 | 34.570 | 7.3× |
| 1024² | 3 565 | 61 | 60.300 | 8.919 | 176.647 | 6.8× |
| 2048² | 13 939 | 53 | 255.386 | 32.120 | 1 160.221 | 8.0× |
| 4096² | 55 319 | 57 | 1 091.265 | 114.805 | 8 769.883 | **9.5×** |

### shapes — rectangles, ring, diagonals, one long snake

| Size | Components | Iterations | CPU BFS (ms) | GPU compress (ms) | GPU no-compress (ms) | Speed-up |
|---|---|---|---|---|---|---|
| 256² | 4 | 117 | 0.442 | 3.328 | 6.714 | 0.1× |
| 512² | 13 | 269 | 1.643 | 6.561 | 13.293 | 0.3× |
| 1024² | 53 | 677 | 7.004 | 24.169 | 39.040 | 0.3× |
| 2048² | 201 | 1 089 | 24.334 | 131.437 | 222.235 | 0.2× |
| 4096² | 779 | 1 893 | 90.657 | 822.562 | 1 771.234 | **0.1×** |

### What the sweep actually shows

- **Geometry, not pixel count, sets the cost.** `noise` at 4096² has 55 319 components and converges in 57 iterations; `shapes` at the same size has 779 components and needs 1 893. The GPU is 9.5× faster on the first and 10× *slower* than a single CPU thread on the second.
- **The GPU loses on thin, long components.** A serpentine path has diameter O(n²); min-label propagation needs iterations proportional to that diameter regardless of how many threads are available. `shapes` is the honest counter-example to any blanket "GPU wins" claim, and it is reported here rather than omitted.
- **Path compression is worth far more than it looks on blobs.** On `noise` 4096² it is the difference between 114.8 ms and 8 769.9 ms — a factor of ~76, and 57 iterations instead of 5 067. On `blobs` the factor is ~4. On `shapes` ~2.
- **Blobs peak at 1024² and fall off.** The 4.3× at 1024² drops to 2.2× at 2048². Iteration count roughly doubles over that step (49 → 85) while the per-iteration work quadruples, and at 4096² the kernel is memory-bound on a 64 MB label array.
- **No speed-up is claimed for the current `src/` revision.** The table above does not transfer to it.

### Against the same algorithm run serially

`cpu-prop` is the identical min-label propagation executed on one CPU thread, so this column isolates what parallelism bought — the CPU BFS column compares against a *better serial algorithm* instead.

| Workload (1024²) | CPU BFS (ms) | CPU prop, same algorithm (ms) | GPU compress (ms) | GPU vs BFS | GPU vs same algorithm |
|---|---|---|---|---|---|
| blobs | 17.795 | 369.632 | 4.158 | 4.3× | **88.9×** |
| noise | 60.300 | 1 450.845 | 8.919 | 6.8× | **162.7×** |
| shapes | 7.004 | 703.383 | 24.169 | 0.3× | **29.1×** |

Two things follow. Min-label propagation is a *poor serial algorithm* — 21× slower than BFS on blobs 1024² — and it is the GPU that makes it competitive at all; quoting 88.9× as "the GPU speed-up" would be the dishonest version of this table, because no one would run propagation serially when BFS exists. And even on `shapes`, where the GPU loses to BFS outright, parallelism still buys 29× against the same algorithm — the deficit is the algorithm's, not the hardware's.

Serial propagation also converges in *fewer* iterations than the GPU (blobs 1024²: 40 vs 49; noise: 44 vs 61; shapes: 513 vs 677), because a raster-order serial sweep carries a label transitively across the whole row in one pass while the parallel version advances every label by one neighbour per iteration.

### Warm-up is a first-order effect — discard it

The verbose convergence trace for blobs 1024² shows the first propagation launches costing roughly eight times the steady-state cost:

```
[iter   1] propagate          0.483 ms  changed=1
[iter   2] propagate          0.486 ms  changed=1
[iter   3] propagate          0.473 ms  changed=1
[iter   4] propagate+compress 0.508 ms  changed=1
[iter   5] propagate          0.068 ms  changed=1
...
[iter  49] propagate          0.063 ms  changed=0  => CONVERGED
```

Nsight corroborates it independently: across 49 launches of `propagate_labels` the median is 65 599 ns but the maximum is 490 551 ns — a single launch 7.5× the median, with the standard deviation (115 134 ns) larger than the median itself. The kernel does fixed per-pixel work regardless of iteration, so this is clock ramp-up and cold caches, not algorithmic.

The practical consequence: the same blobs 1024² workload measures **7.637 ms as a cold single run** and **4.158 ms as the minimum of three warm repetitions** — a 1.8× difference that is purely methodological. Any speed-up quoted from a single cold run is inflated by roughly that factor.

### Where the time goes (Nsight Systems, blobs 1024², 49 iterations)

| Kernel | Share of GPU kernel time | Launches | Total (ms) | Median (µs) |
|---|---|---|---|---|
| `propagate_labels` | 82.7 % | 49 | 4.882 | 65.6 |
| `compress_labels` | 7.8 % | 13 | 0.461 | 34.5 |
| `component_stats` | 7.5 % | 1 | 0.441 | 440.7 |
| `compact_labels` | 0.9 % | 1 | 0.056 | 55.7 |
| `assign_ids` | 0.5 % | 1 | 0.032 | 31.9 |
| `init_labels` | 0.5 % | 1 | 0.030 | 29.9 |

Total GPU kernel time 5.902 ms. The 13 `compress_labels` launches are exactly `ceil(49 / 4)` — the compression interval behaving as configured. Propagation dominates, as the iteration-count analysis predicts, and statistics is a single launch even with global-atomic accumulation. Optimisation effort belongs in the propagation loop — reducing *iterations*, not shaving per-launch cost.

### Choosing the compression interval

Run on `shapes` 1024² (53 components, the workload where chains are longest):

| Setting | Iterations | Time (ms) |
|---|---|---|
| `--compress-every 1` | 676 | 37.530 |
| `--compress-every 4` (default) | 677 | 37.956 |
| `--no-compress` | 1 182 | 59.205 |

Compressing every iteration saves exactly one iteration over compressing every fourth, for four times the compression launches — the benefit is essentially all captured at interval 4, which is why that is the default. Dropping compression entirely costs 1.75× the iterations and 1.58× the time. All three runs produced 53 components and passed verification, so the interval is a pure performance knob with no effect on the answer.

### Convergence behaviour — the dominant performance factor

Double-buffered propagation is the synchronous parallel model, so these sweep counts are hardware-independent, not an approximation:

| Image kind (n × n) | n = 64 | n = 128 | n = 256 | Behaviour |
|---|---|---|---|---|
| blobs (compact regions) | 47 | 76 | 169 | grows roughly with component diameter |
| noise (scattered) | 47 | 92 | 107 | nearly flat — best case for the GPU |
| spiral (one long path) | 125 | 253 | 509 | about 2n: long but shallow |
| snake (one serpentine path) | 2 016 | 8 128 | 32 640 | exactly n²/2 − n/2 — use `--algo uf` |

This is why `uf` is the fast path and `dbuf` is a cross-check rather than the production algorithm.

---

## Repository structure

```
gpu-assignment/
├── src/
│   ├── ccl_core.h             # Per-pixel bodies as __host__ __device__, atomics, union-find
│   ├── ccl_kernels.cuh        # All 14 kernels, thin wrappers over ccl_core.h
│   ├── ccl_cuda.cu            # Host driver: streams, convergence loop, prefix sum, CLI
│   ├── ccl_sim.cpp            # Serial twin — same bodies, permutable visit order
│   ├── ccl_cpu.c              # Independent C99 reference: BFS + serial propagation
│   └── ccl_io.h               # Binary image / label I/O and the statistics CSV writer
├── python/
│   ├── gen_image.py           # 13 deterministic image kinds
│   ├── ccl_io.py              # I/O, canonical relabelling, statistics
│   ├── ccl_numpy.py           # NumPy/SciPy reference
│   ├── verify.py              # Partition, byte-equality and statistics checking
│   ├── benchmark.py           # Scope-aware CPU/GPU timing sweep
│   ├── plot_results.py        # Time / speed-up plots
│   ├── visualize.py           # Component visualization
│   └── parse_colab_output.py  # Benchmark summarization
├── tests/
│   ├── run_tests.py           # The verification matrix
│   ├── nvrtc_check.py         # Device compile with no nvcc
│   └── host_check.sh          # Host type-check with no CUDA toolkit (+ cudashim/)
├── Makefile
├── build_windows.bat
└── README.md
```

The separate coursework archive additionally holds the report, handbook, presentation, narrated video, Colab notebook, browser demo and results. Those assets are not all present in this repository, so this README does not link to repository paths that do not exist.

---

## Kernels

| Kernel | Launch | Role |
|---|---|---|
| `k_init` | 1-D, 256 | `labels[i] = img[i] ? i+1 : 0` |
| `k_prop_inplace` | 2-D, 32×8 | In-place 8-neighbour minimum propagation (`--algo prop`) |
| `k_prop_dbuf` | 2-D, 32×8 | Double-buffered propagation, read-one/write-other (`--algo dbuf`) |
| `k_compress` | 1-D, 256 | Pointer-jumping: replace a label by its current root |
| `k_uf_reduce` | 2-D, 32×8 | Union-find over the W/NW/N/NE edges (`--algo uf`) |
| `k_mark_roots` | 1-D, 256 | `flags[i] = 1` where `labels[i] == i+1` |
| `k_scan_block`, `k_add_offsets` | 1-D, 1024 + shared | Per-block Hillis-Steele exclusive scan, then block offsets added back |
| `k_assign_ids`, `k_relabel` | 1-D, 256 | Compact ids `1..K` via each root's scan slot |
| `k_stats_init` | 1-D, 256 over K+1 | Resets area, coordinate sums, bounding box |
| `k_stats_accum` | 2-D, 32×8 | `atomicAdd` area and 64-bit sums; `atomicMin`/`atomicMax` bbox |
| `k_centroid` | 1-D, 256 over K+1 | `cx = sum_x/area`, `cy = sum_y/area` |
| `k_count_diff` | 1-D, 256 | On-device label-image comparison used by `--cross-check` |

Blocks are 256 threads throughout; 2-D kernels use a 32×8 tile so a warp covers 32 contiguous pixels of one row and global loads coalesce. Thrust and CUB are deliberately not used, so the prefix sum is our own GPU work.

The Colab lab's six kernels map onto this as `init_labels` → `k_init`, `propagate_labels` → `k_prop_inplace`, `compress_labels` → `k_compress`, `assign_ids` + `compact_labels` → the scan-based compaction stage, and `component_stats` → `k_stats_*` fused into one launch. It uses a square 16×16 block rather than a 32×8 tile; both are 256 threads, but only the 32×8 shape gives a warp 32 contiguous pixels of a single row.

---

## Testing, results and evidence

| Evidence | Recorded finding | Interpretation |
|---|---|---|
| CPU and serial-twin matrix, 41 images × 12 configurations | 480 / 480 pass | Reproducible on any machine, no GPU required |
| GPU-inclusive matrix, 41 images × 16 configurations (previous revision, T4) | 636 / 636 pass | Device-inclusive; the current revision has not been re-run |
| Repeat determinism, 4096×4096 (previous revision, T4) | 20 / 20 bit-identical | Races often appear only at scale |
| Order invariance, 4 image kinds × 2 algorithms × 12 visit orders | 96 runs → 1 distinct label image | Iteration count varied, the result did not |
| 1100×1000 image (three levels of scan recursion) | 355 components, 0 differing pixels, statistics exact | Crosses the 1024-element scan block size |
| Known-geometry canvas, 4 rectangles + 5 discs | 9 / 9 areas, centroids and boxes exact | Predicted in closed form before measurement, SciPy-independent |
| Colab lab vs CPU BFS, 1024² blobs | 250 = 250 components, partition identical, 0 of 250 statistics mismatches | Legacy revision, on a T4 |
| Colab lab vs SciPy `ndimage.label`, 1024² blobs | 250 = 250 components, partition identical, 0 mismatches | Independent third implementation |
| Colab lab, `shapes` 1024², three compression settings | 53 components in all three, verify PASS | Compression changes iteration count, never the answer |
| Colab lab, GPU-side convergence trace | `changed=1` for 48 iterations, `changed=0` at 49 → CONVERGED | The 4-byte flag works as designed |
| T4 benchmark sweep, 3 kinds × 5 sizes × 3 reps | See [Measured results](#measured-results--tesla-t4) | Legacy revision only |
| T4 timing, current revision | Pending | No validated device benchmark is claimed |

Two recorded CPU baselines from different containers disagree by ~40 % on the same workload (2.831 ms vs ≈2.0 ms for 512² blobs) — itself the argument for re-measuring on the submission machine rather than quoting either.

### Sample statistics output (blobs 1024², ten largest components)

Written by the GPU and checked pixel-for-pixel against the CPU reference and SciPy:

| id | area | cx | cy | xmin | ymin | xmax | ymax |
|---|---|---|---|---|---|---|---|
| 114 | 13 523 | 872.723 | 499.507 | 756 | 429 | 986 | 582 |
| 147 | 8 316 | 125.339 | 675.611 | 69 | 606 | 173 | 742 |
| 97 | 7 907 | 111.064 | 402.106 | 32 | 357 | 191 | 449 |
| 84 | 7 517 | 621.692 | 338.383 | 550 | 300 | 692 | 395 |
| 140 | 6 923 | 974.148 | 661.307 | 907 | 581 | 1023 | 730 |

Component 140 touches `xmax = 1023`, the last column — a boundary component carried through labelling and statistics without special-casing.

### Edge cases with analytically derived answers

K is known by hand for these, so the test does not depend on SciPy. All matched across `cpu/bfs`, `prop`, `dbuf` and `uf`.

| Image | K | Reasoning |
|---|---|---|
| empty 64×64, empty 1×1 | 0 | no foreground; also the degenerate single-pixel image |
| full 64×64 | 1 | every pixel foreground |
| single 64×64 | 1 | one pixel at linear index 0 — checks that label 0 stays reserved |
| corners 64×64 | 4 | four isolated corner pixels |
| border 64×64 | 1 | the frame is one closed loop |
| checker 64×64 | 1 | diagonal-only adjacency; 8-connectivity joins all of it |
| snake 64×64 | 1 | a single serpentine path threading the image |
| diag 64×64 / 48×48 | 22 / 16 | chains every 3 columns → ⌈64/3⌉, ⌈48/3⌉ |

The 41 test images also cover one-row and one-column images, very sparse and very dense noise, odd and non-square dimensions, and a 1025×3 image that crosses the prefix sum's 1024-element block size.

Run on the T4 in the Colab lab, the four edge cases behaved as follows:

| Image | Result | Iterations |
|---|---|---|
| `full` 64×64, all foreground | K = 1, verify PASS | 5 |
| `column` 1×100, single pixel column | K = 1, verify PASS | 5 |
| `diag` 200×200, `np.eye` | K = 1, verify PASS — 8-connectivity joins the whole diagonal | 5 |
| `empty` 64×64, all background | K = 0 — correct, but the legacy `verify.py` raises | 1 |

The 8-connectivity contrast is worth stating plainly: a 48×48 image of diagonal chains is 408 components under 4-connectivity and 16 under the 8-connectivity implemented here.

### Reconciling the report table with the notebook

The project report's §8.2 performance table and the notebook CSVs come from **different T4 sessions** and do not agree:

| Workload | Report §8.2 | Notebook `benchmark_T4_summary.csv` |
|---|---|---|
| blobs 1024² | 18.222 / 3.658 ms → 5.0× | 17.795 / 4.158 ms → 4.3× |
| blobs 4096² | 262.607 / 138.798 ms → 1.9× | 315.356 / 129.479 ms → 2.4× (161 iterations) |
| blobs 4096² iterations | 164 | 161 |
| noise 4096² | 1 156.805 / 115.510 ms → 10.0× | 1 091.265 / 114.805 ms → 9.5× |

The spread is consistent with ordinary Colab run-to-run variation on shared hardware, and the GPU column is stable (115.5 vs 114.8 ms on noise 4096²) while the CPU column moves most. This README quotes the notebook CSVs throughout, because those are the files shipped in the submission archive and reproduced by the attached run. **Before submission, either regenerate the report table from the same CSVs or state in the report which session each table came from.** The report also lists "CUDA Toolkit 11" in §5 while the recorded run compiled under CUDA 13.0 — worth correcting.

---

## What is novel here

1. **Byte-equality cross-validation instead of partition equivalence.** Because the fixed point is unique, three independent algorithms plus a raster-order CPU BFS must agree to the byte — `memcmp`, not a relabelling-tolerant comparison.
2. **A serial twin that compiles the same source lines as the kernels.** `ccl_sim` includes `ccl_core.h` and calls the identical bodies, so an algorithmic bug is reproducible with no GPU, and `--order shuffle` turns order-independence into an empirical test.
3. **Deterministic prefix-sum compaction** rather than an atomic counter — which is what makes the two points above possible at all.
4. **A GPU-free static verification pipeline** — NVRTC device compilation, clang host-only type-checking, and a PTX audit confirming `atom.global.min.s32` and `atom.global.add.u64` were actually emitted.
5. **Scope-separated timing with no mixed-scope speed-ups**, a published workload (`shapes`) on which the GPU loses, and a same-algorithm serial column that separates "parallelism helped" from "we picked a weaker baseline".

---

## Known issues

- **Two `initcheck` errors** from the previous revision's T4 run are unresolved — most likely in the device/host copy buffers around the cross-check path. Passing `racecheck` alone does not rule out all global-memory concurrency problems.
- **The current revision has no device timing.** The benchmark tables above belong to the legacy Colab revision and do not transfer.
- **`verify.py` crashes on an all-background image** in the legacy lab: `partition_equal` calls `pb.max()` on a zero-size foreground array and raises `ValueError: zero-size array to reduction operation maximum which has no identity`. The labeller itself is correct (K = 0 in 1 iteration); the checker needs an early return for the empty-foreground case. This is the only non-PASS in the edge-case run.
- **The report and the notebook disagree on the T4 table** — see [Reconciling the report table](#reconciling-the-report-table-with-the-notebook). One of the two must be regenerated before submission.
- **The optional `CCL_STRICT_ATOMICS` (libcu++) build is not proven to compile**; it needs the toolkit's CCCL headers. Run `make cuda-strict` where they exist and record the result separately.
- **Maximum image size is 2³¹−2 pixels** because labels are 1-based `int32` indices; 64-bit `atomicAdd` for statistics requires `sm_60` or newer (the program warns at startup on older devices).
- **Statistics use global atomics with no shared-memory privatisation**, so an image with few very large components serialises on a handful of addresses.
- **No block-based (2×2) 8-connectivity optimisation** is implemented; the literature shows this is what separates competitive 8-way GPU CCL from the plain pixel-based form.
- **`--cross-check` triples the work** and must be off when benchmarking.

---

## Next steps

1. Re-run the current revision on a Tesla T4: `make test-cuda`, `--cross-check`, the 20× determinism test, and `make sanitize` with a clean exit status.
2. Resolve or document the two `initcheck` findings.
3. Regenerate benchmark CSVs and plots for the current revision with warm-up discarded and five repetitions, using `end_to_end_ms` for the headline comparison, and replace the legacy tables above. The 1.8× cold-vs-warm gap measured here is the reason warm-up discipline is not optional.
4. Fix the empty-foreground case in `verify.py`.
5. Regenerate the report's §8.2 table from the shipped CSVs, and correct the CUDA version in §5.
6. Run the block-size sweep (16×16 against 32×8, and the 1-D block size), reading occupancy from Nsight Compute rather than inferring it from timings.
7. Publish verification logs, labelled-image screenshots and benchmark CSV/plots under `docs/` or `results/`.
8. Implement Playne-Equivalence or block-based union-find and report the accuracy/performance trade-off — the `shapes` column is the workload to beat.

---

## Team contributions

| Contributor | Area | Specific artefacts |
|---|---|---|
| Ritesh Maury (G25AIT2087) | CUDA label propagation, device-side convergence, path compression, repository ownership | `src/ccl_kernels.cuh`; convergence loop in `src/ccl_cuda.cu`; Makefile CUDA targets; repository and README |
| Shrikrishna Tripathi (G25AIT2103) | Component statistics, benchmarking methodology, visualisation | statistics stage in `src/ccl_cuda.cu` and the bodies in `src/ccl_core.h`; `python/benchmark.py`, `plot_results.py`, `visualize.py`; figures |
| Alok Sahni (G25AIT2138) | Algorithm study and literature survey, CPU reference, verification harness, documentation | `src/ccl_cpu.c`; `src/ccl_sim.cpp`; `tests/run_tests.py`, `nvrtc_check.py`, `host_check.sh`; `python/verify.py`; project report |

Integration, the build system, testing, the presentation and viva preparation were shared. The table indicates primary ownership only.

**AI-assisted work.** Generative AI tools were used for conceptual clarification, debugging suggestions, code review and documentation assistance. All design decisions, integration, compilation, execution, testing, validation and interpretation of results were carried out by the team. Each contributor should confirm this statement against their own artefacts before final attribution.

---

## Contributing

Open an issue explaining the problem, environment, reproduction steps and expected behaviour. Changes go on a separate branch with a concise pull request, test cases, and evidence of any measured speed-up — naming the timing scope. Maintainers must respect course and institutional sharing rules.

## License and usage

No open-source license is confirmed for this repository. Public visibility alone does not grant reuse rights. Before adding a LICENSE file, the contributors should agree on authorship and verify that the coursework materials and third-party assets can legally be redistributed. Until then, request permission before copying or redistributing the code.

## References

1. Rosenfeld & Pfaltz, "Sequential operations in digital picture processing," *JACM* 13(4), 1966.
2. Hawick, Leist & Playne, "Parallel graph component labelling with GPUs and CUDA," *Parallel Computing* 36(12), 2010.
3. Kalentev, Rai, Kemnitz & Schneider, "Connected component labeling on a 2D grid using CUDA," *JPDC* 71(4), 2011.
4. Štava & Beneš, "Connected component labeling in CUDA," in *GPU Computing Gems*, ch. 35, 2011.
5. Playne & Hawick, "A new algorithm for parallel connected-component labelling on GPUs," *IEEE TPDS* 29(6), 2018.
6. Allegretti, Bolelli, Cancilla & Grana, "Optimizing GPU-based connected components labeling algorithms," *IPAS*, 2018.
7. Harris, Sengupta & Owens, "Parallel prefix sum (scan) with CUDA," in *GPU Gems 3*, ch. 39, 2007.
8. He, Chao & Suzuki, "A run-based two-scan labeling algorithm," *IEEE Trans. Image Processing*, 2017.
9. Gonzalez & Woods, *Digital Image Processing*, 4th ed., Pearson, 2018.
10. NVIDIA, *CUDA C++ Programming Guide*.
11. SciPy `ndimage.label` documentation.

---

*Academic project: GPU Programming, Problem 3, IIT Jodhpur. Hardware-performance claims for the current revision remain subject to native-device validation.*
