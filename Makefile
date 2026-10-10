# GPU-Based Connected-Component Labeling — build
ARCH ?= sm_75                # T4 = sm_75, RTX 30xx = sm_86, RTX 40xx = sm_89
NVCC ?= nvcc
CC   ?= gcc

all: ccl_cuda ccl_cpu

ccl_cuda: src/ccl_cuda.cu
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

ccl_cpu: src/ccl_cpu.c
	$(CC) -O3 -o $@ $<

demo: all
	python3 python/gen_image.py blobs.bin 1024 1024 --kind blobs --png blobs.png
	./ccl_cuda --image blobs.bin --out-labels l_gpu.bin --out-stats s_gpu.csv --verbose
	./ccl_cpu  --image blobs.bin --out-labels l_cpu.bin --out-stats s_cpu.csv
	python3 python/verify.py blobs.bin l_gpu.bin s_gpu.csv --ref-labels l_cpu.bin
	python3 python/visualize.py blobs.bin l_gpu.bin s_gpu.csv fig1_components.png

bench: all
	python3 python/benchmark.py --exe ./ccl_cuda --cpu ./ccl_cpu --out results/benchmark.csv
	python3 python/plot_results.py results/benchmark.csv figures/fig5_gpu_benchmark.png

clean:
	rm -f ccl_cuda ccl_cpu *.bin *.csv *.png
