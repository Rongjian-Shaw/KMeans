# gcc -O3 -std=gnu99 -mfma -mavx2 -mavx kernels.c -o kernels.x
gcc -O3 -std=gnu99 -mfma -mavx2 -mavx kernels.c -o kernels.x -fopenmp
