#include <stdio.h>
#include <time.h>

int main() {
    int side = 128;
    int T = 10000;

    clock_t time = - clock();
    float * solution = new float[2 * side * side];

    for (int i=0; i<side; ++i) {
        solution[i] = 1.;
        solution[i + side * side] =  1.;
    }
    
    for (int i=1; i<side; ++i) {
        for (int j=0; j<side; ++j) {
            solution[i * side + j] = 0.;
            solution[i * side + j + side * side] = 0.;
        }
    }

    int half_0, half_1;
    for (int t=0; t<T; ++t) { 
        half_0 = t % 2 * side * side;
        half_1 = (t + 1) % 2 * side * side;
        for (int i=1; i<side-1; ++i) {
            for (int j=1; j<side-1; ++j) {
                solution[i * side + j + half_1] 
                    = solution[(i - 1) * side + j + half_0]
                    + solution[(i + 1) * side + j + half_0]
                    + solution[i * side + j - 1 + half_0]
                    + solution[i * side + j + 1 + half_0];
                solution[i * side + j + half_1] *= .25;
            }
        }
    }
    time += clock();
    printf("elapsed time: %f s\n", (double)time / CLOCKS_PER_SEC);

    FILE * cfout = fopen("output.bin", "wb");
    fwrite(solution, sizeof(float), side * side, cfout);
    fclose(cfout);

    delete [] solution;
}
