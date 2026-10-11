GPU Assingnment
# GPU-Based Connected-Component Labeling — Final Submission
**8-connectivity · iterative label propagation · convergence detected on the GPU · area / centroid / bounding box with atomics**
PGD AI, IIT Jodhpur · GPU Programming · Problem 3
Ritesh Maury (G25AIT2087) · Shrikrishna Tripathi (G25AIT2103) · Alok Sahni (G25AIT2138)

| Folder | Contents | Open with |
|---|---|---|
| 01_Report | 6-page project report (pdf + docx) | any PDF viewer / Word |
| 02_Handbook | Submission handbook: install, run, demo, troubleshoot, full source listing (pdf + docx) | PDF viewer / Word |
| 03_Presentation | 18-slide deck (pptx + pdf) | PowerPoint |
| 04_Video | Narrated, captioned 1080p walkthrough of all 18 slides (`GPU_CCL_Presentation_Video.mp4`, 4 min 49 s) + narration script and build tool | any video player |
| 05_Source_Code | `src/ccl_cuda.cu`, `src/ccl_cpu.c`, Python tools, Makefile, build_windows.bat | nvcc / gcc / MSVC |
| 06_Live_Demo | Browser model of the CUDA pipeline with live kernel log — **double-click, no GPU needed** | any browser |
| 07_Colab_Notebook | Live CUDA lab — upload to Google Colab, Runtime → T4 GPU → Run all | Google Colab |
| 08_Results_and_Figures | Measured CPU baseline CSV, sample statistics CSV, figures; T4 benchmark CSV template | Excel / image viewer |
| 09_Guides | Live demo script, lab guide, viva Q&A, platform setup, README | any text editor |

Key facts: only **4 bytes** cross the PCIe bus per iteration (`d_changed`); the number of components K is counted on the device
with an atomic counter; area, centroid and bounding box are one `component_stats` kernel launch using `atomicAdd / atomicMin / atomicMax`.
CPU baseline (single core, gcc -O3) is measured and included; GPU labels + statistics are validated against the CPU reference and
`scipy.ndimage.label` by `verify.py` (identical partition, exact statistics).

**Still to fill in before submission:** run `07_Colab_Notebook` on a T4, download `benchmark_T4.csv`, `benchmark_T4_summary.csv`,
`fig5_gpu_benchmark.png` into `08_Results_and_Figures/`, and paste the GPU column + speed-ups into the report table (Section 8.2);
confirm the contribution split.
