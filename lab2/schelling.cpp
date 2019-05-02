#include <random>
#include <algorithm>
#include "mpi.h"

std::random_device seed;
std::mt19937 gen{seed()};
using uid = std::uniform_int_distribution<int>;

class District {
    int size_x;
    int x_max;

    int size_y;
    int y_max;

    float tol;

    int * houses;
    int * intent;

    int count;
    int white;
    int * mask;

    int si(int, int); // safe index

    public:
      int * moving;

      District(int, int, int, int, float);
      ~District();
        
      void initialize(int *, int *);
      void zeroIntent();
      void allLookAround();
      void quickLookFrom(int, int);
      std::pair<int,int> findMoving();
      void settle(int);
};


District::District(int side_x, int x_sup, 
                   int side_y, int y_sup, 
                   float coeff) {
    count = 0;
    white = 0;

    tol = coeff;

    size_x = side_x;
    size_y = side_y;

    x_max  = x_sup;
    y_max  = y_sup;

    mask   = (int *)malloc(size_x * size_y * sizeof(int));
    moving = (int *)malloc(size_x * size_y * sizeof(int));
}

District::~District() {
    free(mask);
    free(moving);
}

int District::si(int i, int max) {
    if (i < 0)
        return max - 1;
    else if (i > max)
        return 0;
    return i;
}

void District::initialize
(int * huts, int * will) {
    houses = huts;
    intent = will;

    std::uniform_int_distribution<int> D(0, 1);
    for (int i=0; i<size_y; ++i) {
        for (int j=0; j<size_x; ++j)
            houses[i * size_x + j] = D(gen);
    }
}

void District::zeroIntent() {
    for (int i=0; i<size_y; ++i) {
        for (int j=0; j<size_x; ++j) 
            intent[i * size_x + j] = 0;
    }
}

void District::quickLookFrom(int i, int j) {
    int match1, match2, match3; 
    match1 = std::fabs(houses[i * x_max + j] 
           - houses[i * x_max + si(j+1, x_max)]);

    match2 = std::fabs(houses[i * x_max + j] 
           - houses[si(i-1, y_max) * x_max + j]);

    match3 = std::fabs(houses[i * x_max + j] 
           - houses[si(i-1, y_max) * x_max + si(j+1, x_max)]);

    intent[i * x_max + j] += match1; 
    intent[i * x_max + si(j+1, x_max)] += match1; 

    intent[i * x_max + j] += match2; 
    intent[si(i-1, y_max) * x_max + j] += match2; 

    intent[i * x_max + j] += match3; 
    intent[si(i-1, y_max) * x_max + si(j+1, x_max)] += match3; 
}

void District::allLookAround() {
    for (int i=0; i<size_y; ++i) {
        for (int j=0; j<size_x; ++j)
            quickLookFrom(i,j); 
    } 
}

std::pair<int,int> District::findMoving() {
    int address;
    for (int i=0; i<size_y; ++i) {
        for (int j=0; j<size_x; ++j) {
            address = i * size_x + j;
            if ((double)intent[address] > 8 * tol) {
                white += houses[address];
                moving[count] = address;
                count++;
            }
        } 
    }
    return std::make_pair(count, white);
}

void District::settle(int white) {
    for (int i=0; i<white; ++i) 
        mask[i] = 1;
    for (int i=white; i<count; ++i)
        mask[i] = 0;

    std::shuffle(mask, mask+count, gen);
    for (int i=0; i<count; ++i)
        houses[moving[i]] = mask[i];

    count = 0;
}



int main(int argc, char ** argv) {
    int size_x(200), size_y(20), 
         x_max(200), y_max(200);
    int n_steps = std::atoi(argv[1]);
    float coeff = std::atof(argv[2]);

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int * houses;
    int * intent;
    MPI_Win win;

    MPI_Win_allocate(x_max * y_max * sizeof(int), sizeof(int), 
                          MPI_INFO_NULL, comm, &houses, &win);
    MPI_Win_allocate(x_max * y_max * sizeof(int), sizeof(int), 
                          MPI_INFO_NULL, comm, &intent, &win);
    int pid, num_proc;
    MPI_Comm_rank(comm, &pid);
    MPI_Comm_size(comm, &num_proc);
    
    District region(size_x, x_max, size_y, y_max, coeff);
    region.initialize(houses + pid * size_y,
                      intent + pid * size_y);
    MPI_Barrier(comm);

    int white, whites;
    int * counts;
    int * decision;

    if (pid == 0) {
        counts   = (int *)malloc(num_proc * sizeof(int));
        decision = (int *) malloc(num_proc * sizeof(int));
    }
    std::pair<int,int>  local_rebels;

    MPI_File fh;
    MPI_File_open(comm, "initial.bin",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, pid * size_x * size_y * sizeof(int),
                      MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
    MPI_File_write(fh, houses + pid * size_y, 
                    size_x * size_y, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    for (int t=0; t<n_steps; ++t) {
        region.zeroIntent();
        MPI_Barrier(comm);
        region.allLookAround();
        local_rebels = region.findMoving();

        MPI_Gather(&local_rebels.first, 1, MPI_INT, 
                        counts, 1, MPI_INT, 0, comm);
        MPI_Reduce(&local_rebels.second, &whites, 1, 
                          MPI_INT, MPI_SUM, 0, comm);
        if (pid == 0) {
           for (int i=0; i<num_proc; ++i) {
                uid D(0, std::min(counts[i], whites));
                decision[i] = D(gen);
                whites -= decision[i];
            }
        }
        MPI_Scatter(decision, 1, MPI_INT, &white, 
                              1, MPI_INT, 0, comm);
        region.settle(white);
    }
    MPI_File_open(comm, "resulting.bin",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, pid * size_x * size_y * sizeof(int),
                      MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
    MPI_File_write(fh, houses + pid * size_y, 
                    size_x * size_y, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    if (pid == 0) {
        free(counts);
        free(decision);
    }
    MPI_Win_free(&win);
    MPI_Finalize();
}
