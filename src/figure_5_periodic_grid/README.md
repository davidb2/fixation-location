You can run `periodic_grid.cc` by first compiling like:
```cpp
g++ \
  -std=c++17 \
  -ffast-math \
  -fopenmp \
  -Wall \
  -Wextra \
  -o periodic_grid \
  periodic_grid.cc 
```
and then run like:
```bash
OMP_NUM_THREADS=16 ./periodic_grid
```

You may need to use `clang++` if using `LLVM` (Mac).