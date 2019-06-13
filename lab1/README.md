# OpenMP

## 1D Maxwell (EM-wave propagation)

- `Maxwell_1D.ipynb` (problem statement, some theoretical issues)
- `maxwell.cpp` (impractical solution of PDEs)
- `Visualization.ipynb` (visualization for this task)

Not so interesting and most probably impractical (the latter is due to the way it was solved)

## PI

- `pi.c` - a good example of using OMP  

---

___A bit of theory___

[Shared and private variables in OMP](https://www.ibm.com/support/knowledgecenter/en/SSLTBW_2.3.0/com.ibm.zos.v2r3.cbcpx01/cuppvars.htm) 

[Pragmas for OMP](https://www.ibm.com/support/knowledgecenter/SSXVZZ_13.1.5/com.ibm.xlcpp1315.lelinux.doc/compiler_ref/tuoptppp.html) and [many other pragmas](https://www.ibm.com/support/knowledgecenter/SSXVZZ_13.1.5/com.ibm.xlcpp1315.lelinux.doc/compiler_ref/pragma_descriptions.html), which might have a little to do with OMP, but may be useful in the future.

## Hadamard Product

- `hadamard.cpp`  

The goal is to implement the Hadamard product of two vectors with OMP+MPI hybrid. This is an easy task if data transfer among processes occurs with `MPI_Send`-`MPI_Recv`. But I'd like to solve it using MPI-3 (RMA concept, to be precise). Currently, the program is not finished. Since nothing forces me to do it, no one knows when it will happen. However, when it happens, the problem will have to be moved to `lab2`.  
