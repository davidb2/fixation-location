Firt run
```bash
mkdir -p bin/
```

You can run `periodic_grid.cc` by first compiling like:
```cpp
g++ \
  -std=c++17 \
  -ffast-math \
  -fopenmp \
  -Wall \
  -Wextra \
  -o bin/periodic_grid \
  periodic_grid.cc 
```
and then run from the main directory (i.e., run `cd ../../` first) like:
```bash
OMP_NUM_THREADS=16 ./src/figure_5_periodic_grid/bin/periodic_grid
```

You may need to use `clang++` if using `LLVM` (Mac).