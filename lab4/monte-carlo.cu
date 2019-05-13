#include <stdio.h>
#include <math.h>
#include <curand.h>

__global__
void halve_and_sum(float * data) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
    uint j = blockIdx.x * blockDim.x + threadIdx.x 
           + gridDim.x * blockDim.x / 2;

    data[i] -= data[j];
}

__global__
void expnegsqr_map(float * data, int vol) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = exp(-vol * data[i] * data[i]);
}

template <uint blockSize>
__device__ void warpReduce(volatile float * sdata, uint tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >=  8) sdata[tid] += sdata[tid + 4];
    if (blockSize >=  4) sdata[tid] += sdata[tid + 2];
    if (blockSize >=  2) sdata[tid] += sdata[tid + 1];
}

template <uint dimSize, uint blockSize>
__global__ void sum_reduction(float * gdata) {
    extern __shared__ float sdata[];
    uint tid = threadIdx.x;
    uint i = blockIdx.x * blockSize * 2 + tid;
    if (dimSize < 2)
        sdata[tid] = gdata[i] + gdata[i + blockSize];
    else
        sdata[tid] = gdata[i];
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



int main(int argc, char ** argv) {
    curandGenerator_t gen; 
    size_t halfside = atoi(argv[1]);
    size_t vol = 4 * halfside * halfside;
    const size_t n_blocks(1 << 8); 
    const size_t n_threads(1 << 10);
    size_t n_samples = n_blocks * n_threads << 2;
    float * data;

    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 

    cudaEventRecord(start);
    //------------------------------
    cudaMallocManaged(&data, n_samples * sizeof(float));
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 123456);

    curandGenerateUniform(gen, data, n_samples);

    halve_and_sum<<<2 * n_blocks, n_threads>>>(data);
    expnegsqr_map<<<2 * n_blocks, n_threads>>>(data, vol);

    sum_reduction<n_blocks, n_threads><<<n_blocks, n_threads, 
                            sizeof(float) * n_threads>>>(data);
    sum_reduction<1, n_blocks><<<1, n_blocks, 
                            sizeof(float) * n_blocks>>>(data);
    cudaDeviceSynchronize();
    //------------------------------
    cudaEventRecord(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("res = %f\n", vol * data[0] / (n_blocks * n_threads));
    printf("elapsed time: %f ms\n", elapsed_time);
    curandDestroyGenerator(gen);
    cudaFree(data);
}
