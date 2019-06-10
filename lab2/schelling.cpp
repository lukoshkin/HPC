// Compile & run (for example) with:
// mpic++ schelling.cpp
// mpirun -n 8 ./a.out 1000 .51

#include <random>
#include <algorithm>
#include "mpi.h"

#define CHECK true

std::random_device seed;
std::mt19937 gen{seed()};
using uid = std::uniform_int_distribution<int>;

class District {
    int x_max;

    int size_y;
    int y_max;
    int y0;

    float tol;

    int * houses;
    int * intent;

    int count;
    int white;
    int * mask;
    int * moving;

    int si(int, int); // safe index

    public:

      District(int, int, int, int, float);
      ~District();
        
      void initialize(int *, int *);
      void zeroIntent();
      void quickLookFrom(int, int);
      void allButBottomLookAround();
      void bottomOnlyLookAround();
      std::pair<int,int> findMoving();
      void settleResidents(int);
      int  conductCensus();
};

District::District(int sup_X, 
                   int Y_size, int sup_Y, 
                   int id,   float coeff) {
    count = 0;
    white = 0;

    tol = coeff;

    size_y = Y_size;

    x_max  = sup_X;
    y_max  = sup_Y;

    y0 = id * size_y;

    mask   = (int *)malloc(x_max * size_y * sizeof(int));
    moving = (int *)malloc(x_max * size_y * sizeof(int));
}

District::~District() {
    free(mask);
    free(moving);
}

int District::si(int i, int max) {
    if (i < 0)
        return max - i;
    else if (i > max - 1)
        return i - max;
    return i;
}

void District::initialize
(int * huts, int * will) {
    houses = huts;
    intent = will;

    uid D(0, 1);
    for (int i=y0; i<y0+size_y; ++i) {
        for (int j=0; j<x_max; ++j)
            houses[i * x_max + j] = D(gen);
    }
}

void District::zeroIntent() {
    for (int i=y0; i<y0+size_y; ++i) {
        for (int j=0; j<x_max; ++j) 
            intent[i * x_max + j] = 0;
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

void District::allButBottomLookAround() {
    for (int i=y0; i<y0+size_y-1; ++i) {
        for (int j=0; j<x_max; ++j)
            quickLookFrom(i,j); 
    } 
}

void District::bottomOnlyLookAround() {
    for (int j=0; j<x_max; ++j)
        quickLookFrom(y0+size_y-1, j); 
}

std::pair<int,int> District::findMoving() {
    int address;
    for (int i=y0; i<y0+size_y; ++i) {
        for (int j=0; j<x_max; ++j) {
            address = i * x_max + j;
            if ((float)intent[address] > 8 * tol) {
                white += houses[address];
                moving[count] = address;
                count++;
            }
        } 
    }
    return std::make_pair(count, white);
}

void District::settleResidents(int ones) {
    for (int i=0; i<ones; ++i) 
        mask[i] = 1;
    for (int i=ones; i<count; ++i)
        mask[i] = 0;

    std::shuffle(mask, mask+count, gen);
    for (int i=0; i<count; ++i)
        houses[moving[i]] = mask[i];

    count = 0;
    white = 0;
}

int District::conductCensus() {
    int ones(0);
    for (int i=y0; i<y0+size_y; ++i) {
        for (int j=0; j<x_max; ++j)
            ones += houses[i * x_max + j];
    }
    return ones;
}

/////////////////////////// MAIN /////////////////////////////////////
int main(int argc, char ** argv) {
    int n_steps = std::atoi(argv[1]);
    float coeff = std::atof(argv[2]);
    MPI_Init(&argc, &argv);

    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 
                                  0, MPI_INFO_NULL, &shmcomm);

    int pid, num_proc, world_num_proc;
    MPI_Comm_rank(shmcomm, &pid);
    MPI_Comm_size(shmcomm, &num_proc);
    MPI_Comm_size(MPI_COMM_WORLD, &world_num_proc);

    if (world_num_proc != num_proc) MPI_Abort(MPI_COMM_WORLD, 1);
    
    int x_max(200), size_y(20);
    int y_max = num_proc * size_y;

    MPI_Win win;
    int * baseptr, * houses, * intent;

    int loc_size = (pid == 0) ? 2 * x_max * y_max : 0;

    MPI_Win_allocate_shared(loc_size * sizeof(int), sizeof(int), 
                          MPI_INFO_NULL, shmcomm, &baseptr, &win);
    if (pid != 0) {
        int disp_unit;
        MPI_Aint win_size;
        MPI_Win_shared_query(win, 0, &win_size, &disp_unit, &baseptr);
    }

    houses = baseptr;
    intent = baseptr + x_max * y_max;
    
    double time, max_time;
    time = - MPI_Wtime();

    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

    District region(x_max, size_y, y_max, pid, coeff);
    region.initialize(houses, intent);

#if CHECK
    int population_census;
    int regpop_census = region.conductCensus();
    MPI_Reduce(&regpop_census, &population_census, 1, 
                      MPI_INT, MPI_SUM, 0, shmcomm);
    if (pid == 0)
        printf("number of whites BEFORE: %i\n", 
                population_census);
#endif

    MPI_File fh;
    MPI_File_open(shmcomm, "initial.bin",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, pid * x_max * size_y * sizeof(int),
                      MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
    MPI_File_write(fh, houses + pid * x_max * size_y, 
                    x_max * size_y, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    int white, sum_white;
    int * counts, * decision;
    std::pair<int,int>  local_rebels;

    if (pid == 0) {
        counts   = (int *)malloc(num_proc * sizeof(int));
        decision = (int *) malloc(num_proc * sizeof(int));
    }
    for (int t=0; t<n_steps; ++t) {
        region.zeroIntent();
        region.allButBottomLookAround();

        MPI_Win_sync(win);
        MPI_Barrier(shmcomm);

        region.bottomOnlyLookAround();
        local_rebels = region.findMoving();

        MPI_Gather(&local_rebels.first, 1, MPI_INT, 
                        counts, 1, MPI_INT, 0, shmcomm);
        MPI_Reduce(&local_rebels.second, &sum_white, 1, 
                          MPI_INT, MPI_SUM, 0, shmcomm);
        if (pid == 0) {
            int C, min, max;
            C = std::accumulate(counts, counts + num_proc, 0);

            //one can also create an array of indices from 0 to num_proc,
            //shuffle it, and substitute the looping over the ordered 
            //sequence with the iteration over the resulting array 
            for (int i=0; i<num_proc; ++i) {
                C -= counts[i];
                max = std::min(counts[i], sum_white);
                min = std::fabs(std::min(0, C - sum_white));
                uid D(min, max);
                
                decision[i] = D(gen);
                sum_white -= decision[i];
            }
        }
        MPI_Scatter(decision, 1, MPI_INT, &white, 
                              1, MPI_INT, 0, shmcomm);
        region.settleResidents(white);
    }
    MPI_Win_unlock_all(win);

#if CHECK
    population_census = 0;
    regpop_census = region.conductCensus();
    MPI_Reduce(&regpop_census, &population_census, 1, 
                      MPI_INT, MPI_SUM, 0, shmcomm);
    if (pid == 0)
        printf("___..__..__..___  AFTER: %i\n", 
                population_census);
#endif

    MPI_File_open(shmcomm, "resulting.bin",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, pid * x_max * size_y * sizeof(int),
                      MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
    MPI_File_write(fh, houses + pid * x_max * size_y, 
                    x_max * size_y, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    time += MPI_Wtime();
    MPI_Reduce(&time, &max_time, 1, 
                      MPI_DOUBLE, MPI_MAX, 0, shmcomm);
    if (pid == 0) {
        free(counts);
        free(decision);
        printf("\nElapsed time: %f s\n", time);
    }
    MPI_Win_free(&win);
    MPI_Comm_free(&shmcomm);

    MPI_Finalize();
}
