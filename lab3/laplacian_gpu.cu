#include <stdio.h>

__global__
void laplace(float * U1, float * U2) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int side = blockDim.x + 2;
    U2[(i + 1) * side + j + 1]        // i, j
        = U1[i * side + j + 1]        // i-1, j
        + U1[(i + 1) * side + j]      // i, j-1
        + U1[(i + 2) * side + j + 1]  // i+1, j
        + U1[(i + 1) * side + j + 2]; // i, j+1
    U2[(i + 1) * side + j + 1] *= .25;
}

int main() {
    int T = 10000;
    int side = 128;
    int area = side * side;

    float * U1, * U2, * devU1, * devU2;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  
    //---------------------------
    U1 = (float *)malloc(area * sizeof(float));
    U2 = (float *)malloc(area * sizeof(float));
    cudaMalloc(&devU1, area * sizeof(float));
    cudaMalloc(&devU2, area * sizeof(float));

    for (int i=0; i<side; ++i)
        U1[i] = 1.;

    for (int i=1; i<side; ++i) {
        for (int j=0; j<side; ++j) 
            U1[i * side + j] = 0.;
    }
    memcpy(U2, U1, area * sizeof(float));

    cudaMemcpy(devU1, U1, area * sizeof(float), 
                                 cudaMemcpyHostToDevice);
    cudaMemcpy(devU2, U1, area * sizeof(float), 
                                 cudaMemcpyHostToDevice);

    for (int t=0; t<T;) { 
        laplace<<<side-2, side-2>>>(devU1, devU2);
        laplace<<<side-2, side-2>>>(devU2, devU1);
        t += 2;
    }
    cudaMemcpy(U1, devU1, area * sizeof(float), 
                                 cudaMemcpyDeviceToHost);
    //----------------------------
    cudaEventRecord(stop);
    float elapsed_time(0);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("elapsed time: %f ms\n", elapsed_time);  

    FILE * cfout = fopen("output.bin", "wb");
    fwrite(U1, sizeof(float), area, cfout);
    fclose(cfout);

    cudaFree(devU1);
    cudaFree(devU2);
    free(U1);
    free(U2);
}

