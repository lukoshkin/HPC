# CUDA

What to remember when writing programs on CUDA:

**[Branch Divergence](https://youtu.be/cYw7VsyVTe4)**

- Once a block is assigned to an SM ([streaming multiprocessor](http://www.informit.com/articles/article.aspx?p=2103809)), it is split into several warps of 32 (size depends on the GPU architecture in use)  
- All threads within a *warp* must execute the same instruction at the same time (SIMD-like fashion according to [Flynn's taxonomy](https://en.wikipedia.org/wiki/Flynn%27s_taxonomy#Diagram_comparing_classifications))  
- Branching at the warp scale makes instructions run in serial, and hence, leads to the performance loss - **branch** (or as it's also called, **warp**) **divergence**  
- Keep this in mind, when using `if-else` branches, statements with the ternary operator `p? x: y`, and other kinds within kernels  

**Bank Conflicts**

- The 64k (of shared/L1 memory per SM) is broken into 4 byte sections called *words*  
- Shared memory always reads entire words, regardless of whether you ask for a single byte or a whole word
- To achieve high memory bandwidth for concurrent accesses, shared memory is also organized into larger memory modules (of 32 words), called banks that can be accessed simultaneously
- If multiple addresses of a memory request map to the same memory bank, the accesses are serialized - **bank conflict**  
- If the threads of a warp all request exactly the same value - *no bank conflict*, it is a *broadcast* (and *multicast* when not all but several threads request the same value): the value will be read once and broadcast to the threads

---

In most cases, you can solve these two problems by avoiding them. It is often possible to rearticulate the problem itself or reframe its solution. If not, you may try padded data structures in case of shared memory bank conflicts or to ensure that `if` statements cut on warp boundaries in case of branch divergence.

*Sources:*  
[Post](http://cuda-programming.blogspot.com/2013/02/bank-conflicts-in-shared-memory-in-cuda.html)  
[Video](https://youtu.be/CZgM3DEBplE) - images of *words*, *banks*, examples of padding, and etc. see here

## Monte-Carlo integration with CUDA

This program computes the integral ![equation](https://latex.codecogs.com/gif.latex?%5Cint_%7B-a%7D%5E%7Ba%7D%20%5Cint_%7B-a%7D%5E%7Ba%7D%20e%5E%7B-%28x-y%29%5E2%7Ddxdy) using CURAND library host API

Compile & run with:

```
nvcc monte-carlo.cu -lcurand
./a.out <lim>
```
where `<lim>` is a positive "real" number (`a` symbol in the picture with integral above - sets integration boundaries)  

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

## Visualization

See the visualization for **Histogram** and **Blelloch Scan** in `Visualization.ipynb` 
