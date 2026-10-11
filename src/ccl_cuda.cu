/*
 * ccl_cuda.cu  --  GPU-Based Connected-Component Labeling of a Binary Image (CUDA)
 * PGD AI (IIT Jodhpur) · GPU Programming · Problem 3
 * Ritesh Maury (G25AIT2087) · Shrikrishna Tripathi (G25AIT2103) · Alok Sahni (G25AIT2138)
 *
 * Pipeline (all on the device):
 *   init_labels        : label[i] = i for foreground pixels, 0 for background
 *   propagate_labels   : each foreground pixel pulls the minimum label of its
 *                        8 foreground neighbours; sets a device flag if it changed
 *   compress_labels    : pointer-jumping (path compression) to the root label
 *   compact_labels     : root pixels get consecutive ids 1..K via a device counter
 *   component_stats    : area / centroid sums / bounding box with atomics
 *
 * Convergence is detected ON THE GPU: only one int (4 bytes) is copied back per
 * iteration, never the label image.
 *
 * Build : Linux/Colab : nvcc -O3 -arch=sm_75 -o ccl_cuda ccl_cuda.cu
 *         Windows     : nvcc -O3 -arch=sm_86 -o ccl_cuda.exe ccl_cuda.cu   (x64 Native Tools prompt)
 * Usage : ./ccl_cuda --image img.bin [--out-labels lab.bin] [--out-stats stats.csv]
 *                    [--compress-every 4] [--no-compress] [--verbose]
 *         ./ccl_cuda --gen W H [--density 0.5] [--seed 42] [--dump img.bin] ...
 *
 * Binary image format : int32 W, int32 H, then W*H uint8 (0 = background, 1 = foreground), row-major.
 * Labels output       : int32 W, int32 H, then W*H int32 (0 = background, 1..K = component id).
 * Stats output        : CSV  id,area,cx,cy,xmin,ymin,xmax,ymax
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const int BX = 16, BY = 16;      /* 16x16 = 256 threads per block */

/* ------------------------------------------------------------------------ */
/* Kernel 1: initial labels. One thread per pixel.                          */
/* ------------------------------------------------------------------------ */
__global__ void init_labels(const unsigned char* __restrict__ img,
                            int* __restrict__ labels, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;                       /* boundary guard */
    int idx = y * W + x;
    labels[idx] = img[idx] ? idx : 0;                   /* 0 reserved for background */
}

/* ------------------------------------------------------------------------ */
/* Kernel 2: propagation. Pull the minimum label from the 8-neighbourhood.  */
/* Any change sets *d_changed = 1 (device-side convergence flag).           */
/* ------------------------------------------------------------------------ */
__global__ void propagate_labels(const unsigned char* __restrict__ img,
                                 int* __restrict__ labels, int W, int H,
                                 int* __restrict__ d_changed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    if (!img[idx]) return;                              /* background: never labelled */

    int best = labels[idx];
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        int ny = y + dy;
        if (ny < 0 || ny >= H) continue;                /* top/bottom edge */
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (!dx && !dy) continue;
            int nx = x + dx;
            if (nx < 0 || nx >= W) continue;            /* left/right edge */
            int n = ny * W + nx;
            if (!img[n]) continue;                      /* background neighbour */
            int l = labels[n];
            if (l < best) best = l;
        }
    }
    if (best < labels[idx]) {
        labels[idx] = best;
        atomicExch(d_changed, 1);                       /* flag: not yet converged */
    }
}

/* ------------------------------------------------------------------------ */
/* Kernel 3: path compression (pointer jumping). labels[root] == root.      */
/* ------------------------------------------------------------------------ */
__global__ void compress_labels(const unsigned char* __restrict__ img,
                                int* __restrict__ labels, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    if (!img[idx]) return;
    int l = labels[idx];
    while (labels[l] != l) l = labels[l];
    labels[idx] = l;
}

/* ------------------------------------------------------------------------ */
/* Kernel 4a: every root pixel (labels[i]==i) claims a consecutive id 1..K  */
/* via one atomicAdd on a device counter -> number of components on GPU.   */
/* ------------------------------------------------------------------------ */
__global__ void assign_ids(const unsigned char* __restrict__ img,
                           const int* __restrict__ labels,
                           int* __restrict__ ids, int N, int* __restrict__ counter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (img[i] && labels[i] == i) ids[i] = 1 + atomicAdd(counter, 1);
}

/* Kernel 4b: every foreground pixel takes the id of its root.              */
__global__ void compact_labels(const unsigned char* __restrict__ img,
                               const int* __restrict__ labels,
                               const int* __restrict__ ids,
                               int* __restrict__ out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = img[i] ? ids[labels[i]] : 0;
}

/* ------------------------------------------------------------------------ */
/* Kernel 5: component statistics. One thread per pixel, atomics per label. */
/* ------------------------------------------------------------------------ */
__global__ void component_stats(const int* __restrict__ lab, int W, int H,
                                int* __restrict__ area,
                                unsigned long long* __restrict__ sumx,
                                unsigned long long* __restrict__ sumy,
                                int* __restrict__ xmin, int* __restrict__ ymin,
                                int* __restrict__ xmax, int* __restrict__ ymax)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int k = lab[y * W + x];
    if (k == 0) return;                                 /* background */
    atomicAdd(&area[k], 1);
    atomicAdd(&sumx[k], (unsigned long long)x);
    atomicAdd(&sumy[k], (unsigned long long)y);
    atomicMin(&xmin[k], x); atomicMin(&ymin[k], y);
    atomicMax(&xmax[k], x); atomicMax(&ymax[k], y);
}

/* ------------------------------------------------------------------------ */
/* Host helpers                                                              */
/* ------------------------------------------------------------------------ */
static bool read_img(const std::string& p, std::vector<unsigned char>& img, int& W, int& H)
{
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); return false; }
    if (fread(&W, 4, 1, f) != 1 || fread(&H, 4, 1, f) != 1) { fclose(f); return false; }
    img.resize((size_t)W * H);
    size_t got = fread(img.data(), 1, img.size(), f);
    fclose(f);
    for (auto& v : img) v = v ? 1 : 0;
    return got == img.size();
}
static void write_img(const std::string& p, const std::vector<unsigned char>& img, int W, int H)
{
    FILE* f = fopen(p.c_str(), "wb"); if (!f) return;
    fwrite(&W, 4, 1, f); fwrite(&H, 4, 1, f); fwrite(img.data(), 1, img.size(), f); fclose(f);
}
static void write_labels(const std::string& p, const std::vector<int>& lab, int W, int H)
{
    FILE* f = fopen(p.c_str(), "wb"); if (!f) return;
    fwrite(&W, 4, 1, f); fwrite(&H, 4, 1, f); fwrite(lab.data(), 4, lab.size(), f); fclose(f);
}

/* synthetic image: random blobs (discs + rectangles) on a background, optional noise */
static void gen_image(int W, int H, float density, unsigned seed, std::vector<unsigned char>& img)
{
    std::mt19937 rng(seed);
    img.assign((size_t)W * H, 0);
    if (density >= 0.99f) { for (auto& v : img) v = 1; return; }
    std::uniform_real_distribution<float> u(0.f, 1.f);
    if (density > 0.f && density < 0.35f) {
        /* blob mode: paint random discs until coverage ~ density */
        long target = (long)(density * W * (long)H), painted = 0;
        std::uniform_int_distribution<int> rx(0, W - 1), ry(0, H - 1);
        int rmax = std::max(3, std::min(W, H) / 40);
        std::uniform_int_distribution<int> rr(2, rmax);
        while (painted < target) {
            int cx = rx(rng), cy = ry(rng), r = rr(rng);
            for (int y = std::max(0, cy - r); y <= std::min(H - 1, cy + r); ++y)
                for (int x = std::max(0, cx - r); x <= std::min(W - 1, cx + r); ++x)
                    if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) {
                        unsigned char& v = img[(size_t)y * W + x];
                        if (!v) { v = 1; ++painted; }
                    }
        }
    } else {
        /* noise mode: each pixel foreground with probability = density */
        for (auto& v : img) v = (u(rng) < density) ? 1 : 0;
    }
}

int main(int argc, char** argv)
{
    std::string img_path, out_labels = "labels.bin", out_stats = "stats.csv", dump_path;
    int W = 0, H = 0, compress_every = 4, max_iters = 100000;
    float density = 0.3f; unsigned seed = 42;
    bool gen = false, verbose = false, compress = true;

    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        auto next = [&](const char* what) -> const char* {
            if (a + 1 >= argc) { fprintf(stderr, "missing value for %s\n", what); exit(1); }
            return argv[++a];
        };
        if      (s == "--image")          img_path   = next("--image");
        else if (s == "--out-labels")     out_labels = next("--out-labels");
        else if (s == "--out-stats")      out_stats  = next("--out-stats");
        else if (s == "--dump")           dump_path  = next("--dump");
        else if (s == "--compress-every") compress_every = atoi(next("--compress-every"));
        else if (s == "--no-compress")    compress = false;
        else if (s == "--max-iters")      max_iters = atoi(next("--max-iters"));
        else if (s == "--density")        density = (float)atof(next("--density"));
        else if (s == "--seed")           seed = (unsigned)atoi(next("--seed"));
        else if (s == "--verbose")        verbose = true;
        else if (s == "--gen") { gen = true; W = atoi(next("--gen W")); H = atoi(next("H")); }
        else { fprintf(stderr, "unknown arg %s\n", s.c_str()); return 1; }
    }

    std::vector<unsigned char> h_img;
    if (gen) { gen_image(W, H, density, seed, h_img); if (!dump_path.empty()) write_img(dump_path, h_img, W, H); }
    else if (!read_img(img_path, h_img, W, H)) return 1;
    const int N = W * H;

    int dev = 0; cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    dim3 block(BX, BY), grid((W + BX - 1) / BX, (H + BY - 1) / BY);
    int grid1d = (N + 255) / 256;

    unsigned char* d_img; int *d_lab, *d_ids, *d_out, *d_changed, *d_count;
    CUDA_CHECK(cudaMalloc(&d_img, N));
    CUDA_CHECK(cudaMalloc(&d_lab, (size_t)N * 4));
    CUDA_CHECK(cudaMalloc(&d_ids, (size_t)N * 4));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)N * 4));
    CUDA_CHECK(cudaMalloc(&d_changed, 4));
    CUDA_CHECK(cudaMalloc(&d_count, 4));
    CUDA_CHECK(cudaMemcpy(d_img, h_img.data(), N, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1, eA, eB;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&eA)); CUDA_CHECK(cudaEventCreate(&eB));
    float t, label_ms = 0.f, stats_ms = 0.f;

    if (verbose)
        fprintf(stderr, "[host] gpu=%s SMs=%d | W=%d H=%d N=%d | block=%dx%d grid=%dx%d | compress_every=%d\n",
                prop.name, prop.multiProcessorCount, W, H, N, BX, BY, grid.x, grid.y, compress ? compress_every : 0);

    CUDA_CHECK(cudaEventRecord(e0));
    /* ---- labelling loop ------------------------------------------------ */
    init_labels<<<grid, block>>>(d_img, d_lab, W, H);
    int iters = 0, changed = 1;
    while (changed && iters < max_iters) {
        CUDA_CHECK(cudaMemsetAsync(d_changed, 0, 4));
        CUDA_CHECK(cudaEventRecord(eA));
        propagate_labels<<<grid, block>>>(d_img, d_lab, W, H, d_changed);
        if (compress && ((iters + 1) % compress_every == 0))
            compress_labels<<<grid, block>>>(d_img, d_lab, W, H);
        CUDA_CHECK(cudaEventRecord(eB));
        /* ONLY 4 bytes cross the PCIe bus per iteration */
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventElapsedTime(&t, eA, eB)); label_ms += t;
        ++iters;
        if (verbose) fprintf(stderr, "[iter %3d] propagate%s %.3f ms  changed=%d%s\n", iters,
                             (compress && iters % compress_every == 0) ? "+compress" : "         ", t, changed,
                             changed ? "" : "  => CONVERGED");
    }
    compress_labels<<<grid, block>>>(d_img, d_lab, W, H);       /* final flatten */
    CUDA_CHECK(cudaGetLastError());

    /* ---- compaction to 1..K (K counted on the device) ------------------ */
    CUDA_CHECK(cudaMemset(d_count, 0, 4));
    assign_ids<<<grid1d, 256>>>(d_img, d_lab, d_ids, N, d_count);
    compact_labels<<<grid1d, 256>>>(d_img, d_lab, d_ids, d_out, N);
    int K = 0; CUDA_CHECK(cudaMemcpy(&K, d_count, 4, cudaMemcpyDeviceToHost));

    /* ---- component statistics ----------------------------------------- */
    int *d_area, *d_xmin, *d_ymin, *d_xmax, *d_ymax; unsigned long long *d_sx, *d_sy;
    size_t kb = (size_t)(K + 1) * 4;
    CUDA_CHECK(cudaMalloc(&d_area, kb)); CUDA_CHECK(cudaMalloc(&d_xmin, kb)); CUDA_CHECK(cudaMalloc(&d_ymin, kb));
    CUDA_CHECK(cudaMalloc(&d_xmax, kb)); CUDA_CHECK(cudaMalloc(&d_ymax, kb));
    CUDA_CHECK(cudaMalloc(&d_sx, (K + 1) * 8ull)); CUDA_CHECK(cudaMalloc(&d_sy, (K + 1) * 8ull));
    CUDA_CHECK(cudaMemset(d_area, 0, kb)); CUDA_CHECK(cudaMemset(d_sx, 0, (K + 1) * 8ull)); CUDA_CHECK(cudaMemset(d_sy, 0, (K + 1) * 8ull));
    CUDA_CHECK(cudaMemset(d_xmin, 0x7F, kb)); CUDA_CHECK(cudaMemset(d_ymin, 0x7F, kb));   /* INT_MAX-ish */
    CUDA_CHECK(cudaMemset(d_xmax, 0xFF, kb)); CUDA_CHECK(cudaMemset(d_ymax, 0xFF, kb));   /* -1 */
    CUDA_CHECK(cudaEventRecord(eA));
    component_stats<<<grid, block>>>(d_out, W, H, d_area, d_sx, d_sy, d_xmin, d_ymin, d_xmax, d_ymax);
    CUDA_CHECK(cudaEventRecord(eB)); CUDA_CHECK(cudaEventSynchronize(eB));
    CUDA_CHECK(cudaEventElapsedTime(&stats_ms, eA, eB));
    CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
    float total_ms; CUDA_CHECK(cudaEventElapsedTime(&total_ms, e0, e1));

    /* ---- copy results back once --------------------------------------- */
    std::vector<int> h_lab(N), area(K + 1), xmin(K + 1), ymin(K + 1), xmax(K + 1), ymax(K + 1);
    std::vector<unsigned long long> sx(K + 1), sy(K + 1);
    CUDA_CHECK(cudaMemcpy(h_lab.data(), d_out, (size_t)N * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(area.data(), d_area, kb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sx.data(), d_sx, (K + 1) * 8ull, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sy.data(), d_sy, (K + 1) * 8ull, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xmin.data(), d_xmin, kb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ymin.data(), d_ymin, kb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xmax.data(), d_xmax, kb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ymax.data(), d_ymax, kb, cudaMemcpyDeviceToHost));

    write_labels(out_labels, h_lab, W, H);
    FILE* fs = fopen(out_stats.c_str(), "w");
    if (fs) {
        fprintf(fs, "id,area,cx,cy,xmin,ymin,xmax,ymax\n");
        for (int k = 1; k <= K; ++k)
            fprintf(fs, "%d,%d,%.4f,%.4f,%d,%d,%d,%d\n", k, area[k],
                    (double)sx[k] / area[k], (double)sy[k] / area[k], xmin[k], ymin[k], xmax[k], ymax[k]);
        fclose(fs);
    }
    long fg = 0; for (auto v : h_img) fg += v;
    printf("RESULT W=%d H=%d N=%d fg=%ld components=%d iters=%d time_ms=%.3f label_ms=%.3f stats_ms=%.3f "
           "compress_every=%d gpu=\"%s\"\n", W, H, N, fg, K, iters, total_ms, label_ms, stats_ms,
           compress ? compress_every : 0, prop.name);

    cudaFree(d_img); cudaFree(d_lab); cudaFree(d_ids); cudaFree(d_out); cudaFree(d_changed); cudaFree(d_count);
    cudaFree(d_area); cudaFree(d_sx); cudaFree(d_sy); cudaFree(d_xmin); cudaFree(d_ymin); cudaFree(d_xmax); cudaFree(d_ymax);
    return 0;
}
