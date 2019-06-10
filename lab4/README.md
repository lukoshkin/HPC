# CUDA

## Monte-Carlo integration with CUDA

This program computes the integral ![equation](https://latex.codecogs.com/gif.latex?%5Cint_%7B-a%7D%5E%7Ba%7D%20%5Cint_%7B-a%7D%5E%7Ba%7D%20e%5E%7B-%28x-y%29%5E2%7Ddxdy)

Compile & run with the commands  

```
nvcc monte-carlo.cu -lcurand
./a.out <lim>
```
where `<lim>` is a positive "real" number

Check the correctness of results [here](<https://www.wolframalpha.com/input/?i=int+exp(-(x+-+y)%5E2)+dx+dy,+x+from+-2+to+2,+y+from+-2+to+2>) 

## Histogram with CUDA atomics


## Blelloch Scan
