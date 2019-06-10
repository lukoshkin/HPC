// Compile & run with 
// `nvcc monte-carlo.cu -lcurand`
// `./a.out <lim>`
// (-> integration over `(-lim, lim) x (-lim, lim)`)

// Based on the presentation of Mark Harris:
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

#include <stdio.h>
#include <math.h>
#include <curand.h>

__global__
void halve_and_sum(float * data) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
    uint j = blockIdx.x * blockDim.x + threadIdx.x 
           + gridDim.x  * blockDim.x;

    data[i] -= data[j];
}

__global__
void expnegsqr_map(float * data, int vol) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = exp(-vol * data[i] * data[i]);
}

// Unrolling the last warp
template <uint blockSize>
__device__ void warpReduce(volatile float * sdata, uint tid) {
    // All if-statements are evaluated at compile time
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >=  8) sdata[tid] += sdata[tid + 4];
    if (blockSize >=  4) sdata[tid] += sdata[tid + 2];
    if (blockSize >=  2) sdata[tid] += sdata[tid + 1];
}

// complete unrolling using templates
template <uint folds, uint blockSize>
__global__ void sum_reduction(float * gdata) {
    extern __shared__ float sdata[];
    uint tid = threadIdx.x;

    // All if-statements 
    // but 'internal' ones and the last two
    // are evaluated at compile time

    // multiple adds during load to sh.mem.
    if (folds) {
        sdata[tid] = 0;
        uint i = blockIdx.x * blockDim.x * 2 + tid;
        for (int k=0; k<folds; ++k) {
            sdata[tid] += gdata[i] + gdata[i + blockDim.x];
            i += 2 * blockDim.x * gridDim.x;
        }
    }
    else
        sdata[tid] = gdata[tid];
    __syncthreads();

    if (blockSize >= 1024) { 
        if (tid < 512) sdata[tid] += sdata[tid + 512];
        __syncthreads();
    }
    if (blockSize >=  512) { 
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockSize >=  256) { 
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockSize >=  128) { 
        if (tid <  64) sdata[tid] += sdata[tid +  64];
        __syncthreads();
    }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) gdata[blockIdx.x] = sdata[0];
}



int main(int argc, char * const argv[]) {
    curandGenerator_t gen; 

    float halfside = atof(argv[1]);
    float vol = 4 * halfside * halfside;

    // Ideally, `n_blocks` and `n_threads` 
    // should be chosen based on the architecture 
    // in use to ensure the best performance.
    // If the ideal kernel launch parameters are selected 
    // and it is needed to increase the number of samples,
    // then, one can do it by augmenting `mult`

    const size_t mult = 8; // should be divisible by 2
    const size_t n_blocks(1 << 8); 
    const size_t n_threads(1 << 10);
    size_t n_samples = mult * n_blocks * n_threads;

    float * data;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 

    cudaEventRecord(start);
    //------------------------------
    // n x-s + n y-s = 2 * n samples
    cudaMallocManaged(&data, 2 * n_samples * sizeof(float)); 
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 123456);

    curandGenerateUniform(gen, data, 2 * n_samples);

    halve_and_sum<<<mult * n_blocks, n_threads>>>(data);
    expnegsqr_map<<<mult * n_blocks, n_threads>>>(data, vol);

    sum_reduction<mult / 2, n_threads><<<n_blocks, n_threads, 
                            sizeof(float) * n_threads>>>(data);
    sum_reduction<0, n_blocks><<<1, n_blocks, 
                            sizeof(float) * n_blocks>>>(data);
    //------------------------------
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    printf("exact solution: \t%f\n", 
            2 * halfside * sqrt(acos(-1)) * 
            erf(2 * halfside) - 1 + 
            exp(- 4 * halfside * halfside));

    printf("approx. solution: \t%f\n", 
            vol * data[0] / n_samples);

    printf("number of points sampled: %i\n", n_samples);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("elapsed time: %f ms\n", elapsed_time);

    curandDestroyGenerator(gen);
    cudaFree(data);
}
