// Compile & run with:
// mpic++ naive_transfer.cpp
// mpirun -n 8 ./a.out

#include <cstdio>
#include <mpi.h>

int main (int argc, char *argv[]) {

    int size(10);
    int array[size][size];
    int id, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if (id == 0) {
    //initialize array on the process 0
        for (int i=0; i<size; ++i) {
            for (int j=0; j<size; ++j)
                array[i][j] = 1;
        }
    }

    if (id == 1) {
    //initialize array on the process 1
        for (int i=0; i<size; ++i) {
            for (int j=0; j<size; ++j)
                array[i][j] = 0;
        }
    }

    double _t =- MPI_Wtime();
    if (id == 0) {
        for (int i=0; i<size; ++i)
            MPI_Send (&array[i][0], 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
    }

    if (id == 1) {
        for (int i=0; i<size; ++i)
            MPI_Recv(&array[i][0], 1, MPI_INT, 
                    0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    _t += MPI_Wtime();

    if (id == 1)
        printf("Time spent on data transfer: %f\n", _t);

    if (id == 1) {
        for (int i=0; i<size; ++i) {
            for (int j=0; j<size; ++j)
                printf("%i\t", array[i][j]);
            printf("\n");
        } 
    }

    MPI_Finalize();
}
