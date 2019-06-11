// original code: 
// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

// Compile & run with `nvcc scanBlelloch.cu && ./a.out <x>`
// where `x` defines the lower and upper integral's limits
// i.e. [-x, x] (see the header of `main`)

#include <cmath>
#include <cstdio>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> LOG_NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

__global__ void 
blocksPrescan(float * g_odata, float * g_idata, float * SUMS) {
    extern __shared__ float tmp[];  
    uint N = 2 * blockDim.x;

    uint ai = threadIdx.x;
    uint bi = threadIdx.x + N / 2;

    g_idata += blockIdx.x * N; 
    g_odata += blockIdx.x * N; 

    // load input into shared memory
    uint bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    uint bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    tmp[ai + bankOffsetA] = g_idata[ai]; 
    tmp[bi + bankOffsetB] = g_idata[bi];

    uint offset(1);

    // build sum in place up the tree (up-sweep)
    for (uint d=N>>1; d>0; d>>=1) { 
        __syncthreads();

        if (threadIdx.x < d) {
            uint ai = offset * (2 * threadIdx.x + 1) - 1;
            uint bi = offset * (2 * threadIdx.x + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            tmp[bi] += tmp[ai];
        }
        offset <<= 1;
    }
    // write the sum of the array chunk to 'SUMS'
    // and  clear the last element 
    float t;
    if (!threadIdx.x) {
        uint IDX = N - 1;
        IDX += CONFLICT_FREE_OFFSET(N - 1); 

        if (SUMS) {
            t = tmp[IDX];  
            SUMS[blockIdx.x] = t;
        }
        tmp[IDX] = 0;  
    }

    // traverse down tree & build scan (down-sweep)
    for (uint d=1; d<N; d *= 2) { 
        offset >>= 1;
        __syncthreads();

        if (threadIdx.x < d) {
            uint ai = offset * (2 * threadIdx.x + 1) - 1;
            uint bi = offset * (2 * threadIdx.x + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            t = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += t; 
        }
    }
    __syncthreads();

    // write results to device memory
    g_odata[ai] = tmp[ai + bankOffsetA]; 
    g_odata[bi] = tmp[bi + bankOffsetB];
}


__global__ 
void blocksShifter(float * g_odata, float * SUMS) {
    g_odata += 2 * blockIdx.x * blockDim.x;
    g_odata[threadIdx.x] += SUMS[blockIdx.x];
    g_odata[threadIdx.x + blockDim.x] += SUMS[blockIdx.x];
}

size_t smemSize(int n_el) {
    int extra_space = n_el / NUM_BANKS;
    extra_space += extra_space / NUM_BANKS;
    return sizeof(float) * (n_el + extra_space);
}


//////////////////////// MAIN //////////////////////////////
// Calculation of the integral \int_{-x}^x \exp(- t^2) dt //
////////////////////////////////////////////////////////////
int main(int argc, char ** argv) {
    // set the integral's limits 
    float x = atof(argv[1]);

    // discretization settings
    size_t n_blocks = 512;
    size_t block_size = 2048;
    size_t n_el = n_blocks * block_size;
    printf("Number of discretization points: %i\n", n_el);

    float * idata, * odata, * sums;
    cudaMallocManaged(&idata, n_el * sizeof(float));
    cudaMallocManaged(&odata, n_el * sizeof(float));
    cudaMallocManaged(&sums,  n_blocks * sizeof(float));

    // calculate integrand's values
    float t, dt;
    dt = 2 * x / n_el;
    for (uint i=0; i<n_el; ++i) {
        t = - x + i * dt;
        idata[i] = exp(- t * t) * dt;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // measure the execution time of the framed block
    cudaEventRecord(start);
    //------------------------------
    blocksPrescan<<<n_blocks, block_size / 2, 
                    smemSize(block_size)>>>(odata, idata, sums);

    blocksPrescan<<<1, n_blocks / 2, 
                    smemSize(n_blocks)>>>(sums, sums, NULL);

    blocksShifter<<<n_blocks, block_size / 2>>>(odata, sums);
    //------------------------------
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Elapsed time: %f ms\n", elapsed_time);
    printf("Results are written to 'output.bin'\n");

    FILE * cfout = fopen("output.bin", "wb");
    fwrite(odata, sizeof(float), n_el, cfout);
    fclose(cfout);

    cudaFree(idata);
    cudaFree(odata);
    cudaFree(sums);
}
