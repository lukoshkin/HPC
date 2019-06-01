#include <cstdio>
#include <cmath>
#include <omp.h>
#include <cstdlib>
#define pi 3.141592

int main(int argc, char **argv) {
    int   mult = std::atoi(argv[1]);
    float L = 8 * pi * mult;
    int   N = std::atoi(argv[2]);
    float dx = L / N;
    int   bp = 2 * pi / dx * mult;

// create an array for the solution
    float * F = new float [N * N];

// generate an array containing the boundary values
    float * values = new float [2 * N - 1];

    double t0 = omp_get_wtime();
    #pragma omp parallel 
    {
        #pragma omp for nowait 
        for (int i=0; i<N+bp; ++i) {
            values[i] = std::sin((-N + 1 + i) * dx);
        }
        
        #pragma omp for
        for (int i=N+bp; i<2*N; ++i) {
            values[i] = 0; 
        }
    }
    double t1 = omp_get_wtime();

    printf("time spent on bc initialization : %fs\n", t1 - t0);

// fill the solution array
    #pragma omp parallel for collapse(2) 
    for (int i=0; i<N; ++i) {
        for (int j=0; j<N; ++j) {
            F[j + N * i] = values[i - j + N - 1];
        }
    } 
    printf("time spent on filling the sol. array : %.3fs\n", 
            omp_get_wtime() - t1);

    FILE * fout = fopen("output.bin", "wb");
    fwrite(F, sizeof(float), N * N, fout);
    fclose(fout);

// deallocate memory
    delete [] values;
    delete [] F;
}
