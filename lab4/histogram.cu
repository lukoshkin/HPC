#include <cstdio>
#include <curand_kernel.h>

// uniform integer distrigbution in CUDA:
// https://stackoverflow.com/questions/43622482

// Fewer than 32 bins will take the same amount of time
// because of the warp size. However, it is not possible
// to confirm it in this task, even with fixed seed,
// since the thread execution order can't be fixed
const uint n_bins = 32;
const uint n_blocks = 256;
const uint n_threads = 256;

__global__ 
void setupExperiment(curandState * state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, id, 0, &state[id]);
}

// if curand_uniform is called several times, then generator
// state can be read once and used in local memory, then 
// stored back into global memory (see CURAND documentation)
__global__
void shmemAtomics(curandState * state, uint * buffer) {
    __shared__ uint sdata[n_bins];
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (j < n_bins) sdata[j] = 0;
    int z = i * blockDim.x + j;

    uint sample;
    sample = n_bins * curand_uniform(&state[z]);
    atomicAdd(&(sdata[sample]), 1);
    __syncthreads();

    if (j < n_bins)
        buffer[i * n_bins + j] += sdata[j];
}

// not so many summands to apply sum reduction
__global__ 
void histsPileUp(uint * buffer) {
    uint j = threadIdx.x;
    for (uint i=1; i<n_blocks; ++i)
        buffer[j] += buffer[j + i * n_bins];
}


////////////////////////// MAIN /////////////////////////////////////
int main() {
    curandState * devStates;
    cudaMalloc(&devStates, n_threads * n_blocks * sizeof(curandState));

    uint * hist;
    cudaMallocManaged(&hist, n_blocks * n_bins * sizeof(uint));
    cudaMemset(hist, 0, n_blocks * n_bins * sizeof(uint));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //-------------------------------
    setupExperiment<<<n_blocks, n_threads>>>(devStates);
    for (uint k=0; k<16; ++k) 
        shmemAtomics<<<n_blocks, n_threads>>>(devStates, hist);
    histsPileUp<<<1, n_bins>>>(hist);
    //-------------------------------
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time(0);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Elapsed time: %f ms\n", elapsed_time);
    printf("Results are written to 'output.bin'\n");

    FILE * cfout = fopen("output.bin", "wb");
    fwrite(hist, sizeof(uint), n_bins, cfout);
    fclose(cfout);

    cudaFree(hist);
}
