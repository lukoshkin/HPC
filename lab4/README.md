# Monte-Carlo integration with CUDA
This program computes the integral <img src="https://latex.codecogs.com/gif.latex?\int_{-a}^{a} \int_{-a}^{a} e^{-(x-y)^2}dxdy" />  

Compile and run with the commands  

```
nvcc monte-carlo.cu -lcurand
./a.out 2
```
where instead of `2` you can specify any other positive "real" number

Check the correctness of results [here](<https://www.wolframalpha.com/input/?i=int+exp(-(x+-+y)%5E2)+dx+dy,+x+from+-2+to+2,+y+from+-2+to+2>) (change all `2`s for the number you put as argument to the program)

