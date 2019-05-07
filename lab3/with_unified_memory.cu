#include <stdio.h>

__global__
void laplace(float * U1, float * U2) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    U2[(i + 1) * blockDim.x + j + 1]        // i, j
        = U1[i * blockDim.x + j + 1]        // i-1, j
        + U1[(i + 1) * blockDim.x + j]      // i, j-1
        + U1[(i + 2) * blockDim.x + j + 1]  // i+1, j
        + U1[(i + 1) * blockDim.x + j + 2]; // i, j+1
    U2[(i + 1) * blockDim.x + j + 1] *= .25;
}

int main() {
    int T = 200;
    int side = 128;
    int area = side * side;

    float * U1, * U2;
    cudaMallocManaged(&U1, area * sizeof(float));
    cudaMallocManaged(&U2, area * sizeof(float));

    for (int i=0; i<side; ++i)
        U1[i] = 1.;

    for (int i=1; i<side; ++i) {
        for (int j=0; j<side; ++j) 
            U1[i * side + j] = 0.;
    }
    memcpy(U2, U1, area * sizeof(float));

    for (int t=0; t<T;) { 
        laplace<<<side-2, side-2>>>(U1, U2);
        laplace<<<side-2, side-2>>>(U2, U1);
        t += 2;
    }
    cudaDeviceSynchronize();

    FILE * cfout = fopen("output.bin", "wb");
    fwrite(U1, sizeof(float), area, cfout);
    fclose(cfout);

    cudaFree(U1);
    cudaFree(U2);
}

