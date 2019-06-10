# MPI

## Mini-Project

Basic notions (**communicators, shared memory, independent parallel output**) and common procedures (like **Scatter, Gather, Reduce**) are presented in `schelling.cpp`. Along with `Visualization.ipynb`, these files are one of the possible simulations of Schelling's Segregation Model. Briefly about what this model is. We have two populations, which can be distinguished by color (black and white pixels). A representative (a pixel) of each population can live peacefully in its location while the number of its neighbors of the different color does not exceed a certain threshold. If there are more neighbors than the pixel being considered can tolerate, then it and the other dissatisfied residents (all over the plane, on which the segregation process takes place) are randomly swapped. During the initialization, the residents' (pixels') colors in the rectangular area (which is called "city") are determined randomly. After a number of iteration steps, the segregation process comes to an equilibrium, unless an extreme threshold has been chosen.

***Compile & run with:***
```
mpic++ schelling.cpp
mpirun -n <n_proc> ./a.out <n_iters> <threshold>
```

## Other exercises

There are several seminar's tasks (`naive_transfer.cpp`, `transfer_with_mask.cpp`, `write_to_file.cpp`). From their names it is clear what they solve.
