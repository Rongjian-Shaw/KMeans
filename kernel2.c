/*
Selecting Nearest Centroid

n: number of points
k: number of centroids
dist: n * k distance matrix (double)
mask: n * k mask

After having n * k distance matrix, we want to first find the minimum distance to centroids for each point. 
This would be a n dimensional vector, then we broadcast this vector to match back to the n * k distance matrix to generate 
a n * k mask of 0 or 1 representing if this this the nearest centroid for corresponding point. 
This n * k mask could also be viewed as n k-dimensional one-hot vectors.
 
Input: n * k distance matrix
Output: n * k mask

Independent operation:  Compare distances
Dependent instruction: Compare -> Assign data
Function unit: CMP
*/

// compile: gcc -O3 -std=gnu99 -mfma -mavx2 -mavx kernel2.c -o kernel2.x
// run: kernel2.x <<N: Number points>> 

#include <stdio.h>
#include <stdlib.h>
#include "immintrin.h"
#include <float.h>
#include <string.h>
#include <assert.h>

void print256d(__m256d var)
{
    double val[4];
    memcpy(val, &var, sizeof(val));
    printf("%f %f %f %f \n", val[0], val[1], val[2], val[3]);
}

void transpose16(__m256d* A0, __m256d* A1, __m256d* A2, __m256d* A3) {

    __m256d b0b1d0d1 = _mm256_shuffle_pd(*A0, *A1, 0xf);
    __m256d b2b3d2d3 = _mm256_shuffle_pd(*A2, *A3, 0xf);
    __m256d a0a1c0c1 = _mm256_shuffle_pd(*A0, *A1, 0x0);
    __m256d a2a3c2c3 = _mm256_shuffle_pd(*A2, *A3, 0x0);

    
    *A0 = _mm256_permute2f128_pd(a0a1c0c1, a2a3c2c3, 0x24);
    *A1 = _mm256_permute2f128_pd(b0b1d0d1, b2b3d2d3, 0x24);
    *A2 = _mm256_permute2f128_pd(a2a3c2c3, a0a1c0c1, 0x13);
    *A3 = _mm256_permute2f128_pd(b2b3d2d3, b0b1d0d1, 0x13);

}

void kernel2 (int n, int k, double* dist, double* mask)
{   
    __m256d a0, a1, a2, a3, b0, b1, b2, b3;
    __m256d a, b;
    for (int i = 0; i < n; i += 8) {
        a = _mm256_set_pd(DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
        b = _mm256_set_pd(DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
        for (int j = 0; j < k; j += 4) {
            // load
            a0 = _mm256_load_pd(&dist[i * k + j]);
            a1 = _mm256_load_pd(&dist[(i + 1) * k + j]);
            a2 = _mm256_load_pd(&dist[(i + 2) * k + j]);
            a3 = _mm256_load_pd(&dist[(i + 3) * k + j]);
            b0 = _mm256_load_pd(&dist[(i + 4) * k + j]);
            b1 = _mm256_load_pd(&dist[(i + 5) * k + j]);
            b2 = _mm256_load_pd(&dist[(i + 6) * k + j]);
            b3 = _mm256_load_pd(&dist[(i + 7) * k + j]);

            // transpose
            transpose16(&a0, &a1, &a2, &a3); transpose16(&b0, &b1, &b2, &b3);

            // find min
            a = _mm256_min_pd(a, a0); a = _mm256_min_pd(a, a1); a = _mm256_min_pd(a, a2); a = _mm256_min_pd(a, a3);
            b = _mm256_min_pd(b, b0); b = _mm256_min_pd(b, b1); b = _mm256_min_pd(b, b2); b = _mm256_min_pd(b, b3); 

            // compare and mask
            a0 = _mm256_cmp_pd(a, a0, 0); a1 = _mm256_cmp_pd(a, a1, 0); a2 = _mm256_cmp_pd(a, a2, 0); a3 = _mm256_cmp_pd(a, a3, 0);
            b0 = _mm256_cmp_pd(b, b0, 0); b1 = _mm256_cmp_pd(b, b1, 0); b2 = _mm256_cmp_pd(b, b2, 0); b3 = _mm256_cmp_pd(b, b3, 0);

            // transpose back
            transpose16(&a0, &a1, &a2, &a3); transpose16(&b0, &b1, &b2, &b3);

            // store
            _mm256_store_pd(&mask[i * k + j], a0);
            _mm256_store_pd(&mask[(i + 1) * k + j], a1);
            _mm256_store_pd(&mask[(i + 2) * k + j], a2);
            _mm256_store_pd(&mask[(i + 3) * k + j], a3);
            _mm256_store_pd(&mask[(i + 4) * k + j], b0);
            _mm256_store_pd(&mask[(i + 5) * k + j], b1);
            _mm256_store_pd(&mask[(i + 6) * k + j], b2);
            _mm256_store_pd(&mask[(i + 7) * k + j], b3);
        }
    }
}

int main(int argc, char **argv) {
    int N = 16, K = 4;
    if (argc != 2){
        printf("kernel2.x <<N: Number points>> \n");
        exit(0);
    }  
    else {
        N = atoi(argv[1]);
    }
    
    assert(N % 4 == 0);

    double *dist, *mask;
    posix_memalign((void**) &dist, 64, N * K * sizeof(double));
    posix_memalign((void**) &mask, 64, N * K * sizeof(double));

    for (int i = 0; i != N * K; ++i){
        dist[i] = ((double) rand())/ ((double) RAND_MAX);
    }
    
    printf("Distance Matrix: \n");
    for (int i = 0; i < N; i ++) {
        for (int j = 0; j < K; j ++)
            printf("%f ", dist[i * K + j]);
        printf("\n"); 
    } 
    printf("\n");

    kernel2 (N, K, dist, mask);
    
    printf("Generated Mask: \n");
    for (int i = 0; i < N; i ++) {
        for (int j = 0; j < K; j ++)
            printf("%d ", mask[i * K + j] == 0.0 ? 0 : 1);
        printf("\n"); 
    } 

    free(dist);
    free(mask);
    return 0;
}