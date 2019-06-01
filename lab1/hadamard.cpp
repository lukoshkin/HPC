// THIS PROGRAM IS NOT FINISHED
// Hadamard (element-wise) product of two vectors
// Compile & run with:
// mpic++ hadamard.cpp -fopenmp
// mpirun -n 8 ./a.out'

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <omp.h>
#include <string>

void RandInit(uint size, int * arr) {
    #pragma omp parallel for    
    for (uint i = 0; i < size; ++i)
        arr[i] = rand() % 100;
}

void Print(uint size, int * arr) {
    #pragma omp parallel for
    for (uint i = 0; i < size; ++i) 
        std::cout << std::string(i, '\t') + std::to_string(arr[i]) + '\r';
        // in C++11, std::cout is thread-safe,
        // if `<<` is called only once for a complete action
        // (i.e. it's better avoid .. << str << '\r'.
        //  In this case the 2 calls of `<<` in one thread 
        //  can be interleaved with calls of the other threads)
    std::cout << '\n' << std::string(60, '=') << '\n';
}

int main() {
    uint size = 4;
    srand(time(NULL));

    // UNIMPLEMENTED: write sharing of arrays with RMA
    int a[size];
    int b[size];
    int c[size];

    RandInit(size, a);
    RandInit(size, b);

    Print(size, a);
    Print(size, b);

    int num_proc;
    int pid;
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    #pragma omp parallel for 
    for (uint i=pid; i<size; i+=num_proc)
        c[i] = a[i] * b[i];

    MPI_Finalize();
    Print(size, c);
    printf("The number of threads provided: %i\n", provided);
}

