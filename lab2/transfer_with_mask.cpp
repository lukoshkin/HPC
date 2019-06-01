// Compile & run with:
// mpic++ transfer_with_mask.cpp
// mpirun -n 8 ./a.out

#include <cstdio>
#include <mpi.h>

int main () {
    int size(10);
    int array[size][size];
    int id, num_procs;

    MPI_Init(NULL, NULL);
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

    MPI_Datatype ColumnType;
    MPI_Type_vector (size, 1, size, MPI_INT, &ColumnType);
    MPI_Type_commit(&ColumnType);

    double _t =- MPI_Wtime();
    if (id == 0) {
        MPI_Send(&array[0][0], 1, ColumnType, 1, 1, MPI_COMM_WORLD);
    }

    if (id == 1) {
        MPI_Recv(&array[0][0], 1, ColumnType, 
                 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    _t += MPI_Wtime();

    if (id == 1)
        printf("Time spent on data transfer: %f s\n", _t);

    if (id == 1) {
        for (int i=0; i<size; ++i) {
            for (int j=0; j<size; ++j)
                printf("%i\t", array[i][j]);
            printf("\n");
        } 
    }
    MPI_Finalize();
}
