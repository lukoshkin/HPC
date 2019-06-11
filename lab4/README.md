# CUDA

## Monte-Carlo integration with CUDA

This program computes the integral ![equation](https://latex.codecogs.com/gif.latex?%5Cint_%7B-a%7D%5E%7Ba%7D%20%5Cint_%7B-a%7D%5E%7Ba%7D%20e%5E%7B-%28x-y%29%5E2%7Ddxdy) using CURAND library host API

Compile & run with:

```
nvcc monte-carlo.cu -lcurand
./a.out <lim>
```
where `<lim>` is a positive "real" number

In the implementation of parallel sum reduction I follow the algorithms proposed in [Mark Hariss' presentation](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

## Histogram with CUDA atomics

Compile & run with:

```
nvcc histogram.cu
./a.out
```

---

Partially inspired by [this blog](https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)

The main idea was to try `<cuda_kernel.h>` library (CURAND device API) in practice. So, the data for the histogram are generated within kernel calls. This may not be applicable to real problems.

## Blelloch Scan

The problem which this program solves: making disretization of the function defined as ![equation](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20%5Cint_%7B-x%7D%5Ex%20%5Cexp%28-%20t%5E2%29%20dt)

Based on [this article](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html)

---

Compile & run with:

```
nvcc scanBlelloch.cu
./a.out <x>
```

where `x` defines the lower and upper limits of the integral, i.e. [-x, x]
