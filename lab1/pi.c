// To compile and run, in your shell
// g++ pi.c -fopenmp
// export OMP_NUM_THREADS=32 // or less if you have fewer 
// ./a.out

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand((uint)time(NULL));

    //omp_set_dynamic(0);       // `0/1` - disable/enable dynamic teams
    //omp_set_num_threads(8);   // set 8 threads for all parallel regions 
                                // that follow this command below
    // the second is not valid without the first. `omp_set_dynamic` allows
    // compiler "to vary the degree of parallelism from one parallel region
    // to the next in order to adapt to the system load", i.e. the number of
    // threads is not fixed if it is enabled
   
    uint N_total = 1e6; 
    uint N_in = 0;
    uint counter = 0;
    uint cond;
    float x, y, pi;

    double mo =- omp_get_wtime(); // (!) returns double (I got float overflow once)
    
    // firstprivate(counter) - every thread creates a copy of `counter` 
    //                         (counter's value is also copied)
    // private(x,y)          - every thread creates x, y 
    //                         (if the corresponding indentifiers exist 
    //                         before pragma, in parallel block they appear uninitialized
    // reduction(+:N_in)     - is usually applied to an array, however, in this case
    //                         it also gives the correct result and may be used
    //                         instead of `atomic update`
    #pragma omp parallel firstprivate(counter) private(x,y) //reduction(+:N_in)
    {
        #pragma omp for
        for (uint i=0; i<N_total; ++i) {
            x = (float)rand() / RAND_MAX;
            y = (float)rand() / RAND_MAX;
            cond = (x * x + y * y < 1);  
            counter += cond;
            #pragma omp atomic update
            N_in += cond;
            // N_in is shared by default
            // atomic update - if several threads try to write 
            //                 at a time, the process will be serialized for them.
        }
        printf("counter = %d\n", counter);
    }
    mo += omp_get_wtime();

    printf("sum of counters : %d\n", N_in);
    pi = 4 * (float)N_in / N_total;
    printf("pi = %f\n", pi);
    printf("elapsed time: %f s\n", mo);
}
