// Compile & run with
// mpic++ write_to_file.cpp
// mpirun -n 2 ./a.out
// !it should be exactly 2 nodes!

#include <cstdio>
#include <algorithm>
#include <mpi.h>

// mpi writes data in binary if you use ints instead chars,                      
// so, if you want to get the output in ascii format for ints,                     
// run in your console the following:                                            
//                                                                              
// hexdump -v -e '7/4 "%10d "' -e '"\n"' FILENAME                                
//                                                                              
// or create a buffer array of chars where you will put ints                     
// as chars and transfer for writing (the latter approach is                     
// implemented in this program)
//
// You should take into account that in this implementation 
// MPI writes to file on top of what is already there. So, if 
// the file exists, in some cases you will have to clear file
// before writing 

int main() {
    int id, size(10);
    int array[size];
    char buffer[size];

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if (id == 0) {
        for (int i=0; i<size; ++i)                                            
                array[i] = 0;                                                
        //std::copy(array, array + size, array);
    }
    
    if (id == 1) {
        for (int i=0; i<size; ++i)
                array[i] = 1;                                                
        //std::copy(array, array + size, array);
    } 

    for (int i=0; i<size; ++i)
        buffer[i] = '0' + array[i];
   
    MPI_Status status; // you can pass NULL instead 
                       // if you are not going to use it
    //file handler
    MPI_File fh; 

    //open file, choose what to do with it
    MPI_File_open(MPI_COMM_WORLD, "output.dat", 
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);

    //specify for each process its own section where to write 
    MPI_File_set_view(fh, id * size * sizeof(char), 
                      MPI_CHAR, MPI_CHAR, 
                      "native", MPI_INFO_NULL);

    //each process writes the data it stores in 'array'
    MPI_File_write(fh, buffer, size, MPI_CHAR, &status);

    MPI_File_close(&fh);
    MPI_Finalize();
}
