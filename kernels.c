/**
 * @file kernels.c
 * How to compile: gcc -O3 -std=gnu99 -mfma -mavx2 -mavx kernel3.c -o kernel3.x
 */

#include <assert.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "immintrin.h"

// timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

void print256d(__m256d var);
void transpose16(__m256d* A0, __m256d* A1, __m256d* A2, __m256d* A3);
void maskTransform(int n, int k, double* mask);

/**
 * @brief Calculate the distance between every centroid and data points.
 *
 * @description
 * 1. Traverse every centroid.
 * 2. Get the first 4 elements of the first centroid.
 * 3. Traverse every data point.
 * 4. Get the first 4 elements of every point.
 * 5. Use the first 4 elements of a centroid and the first 4 elements of every
 * point to calculate the first 4 dimensions distance and add it to the dist.
 * 6. Get the second 4 elements, repeat 2~5, until finishing all dimensions of
 * the first centroid.
 * 7. Go to the next centroid, do 2~6, until finishing all of centroids. Then
 * we get dist[n][k]. dist[j][i] means the distance between the j th data point
 * and i th centroid.
 *
 * @param dist n * k distance matrix (double)
 * @param n number of points
 * @param d number of dimensions
 * @param k number of centroids
 */
void kernel1(double* points,
             double* centroids,
             double* dist,
             int n,
             int d,
             int k) {
#pragma omp parallel for num_threads(8)
    for (int c = 0; c < k; c++) {
        for (int kernelHead = 0; kernelHead < d; kernelHead += 4) {
            // [C[0][0], C[0][1], C[0][2], C[0][3]]    C[centroids][d]
            __m256d SIMDC = _mm256_load_pd(&centroids[c * d + kernelHead]);
            for (int i = 0; i < n; i += 4) {
                // [X[0][0], X[0][1], X[0][2], X[0][3]]    X[points][d]
                __m256d SIMDX0 = _mm256_load_pd(&points[i * d + kernelHead]);
                __m256d SIMDX1 =
                    _mm256_load_pd(&points[(i + 1) * d + kernelHead]);
                __m256d SIMDX2 =
                    _mm256_load_pd(&points[(i + 2) * d + kernelHead]);
                __m256d SIMDX3 =
                    _mm256_load_pd(&points[(i + 3) * d + kernelHead]);
                // X - C
                __m256d curDiff0 = _mm256_sub_pd(SIMDX0, SIMDC);
                __m256d curDiff1 = _mm256_sub_pd(SIMDX1, SIMDC);
                __m256d curDiff2 = _mm256_sub_pd(SIMDX2, SIMDC);
                __m256d curDiff3 = _mm256_sub_pd(SIMDX3, SIMDC);
                // (X-C)^2
                __m256d curDist0 = _mm256_mul_pd(curDiff0, curDiff0);
                __m256d curDist1 = _mm256_mul_pd(curDiff1, curDiff1);
                __m256d curDist2 = _mm256_mul_pd(curDiff2, curDiff2);
                __m256d curDist3 = _mm256_mul_pd(curDiff3, curDiff3);
                double tmp0[4];
                double tmp1[4];
                double tmp2[4];
                double tmp3[4];
                _mm256_store_pd(tmp0, curDist0);
                _mm256_store_pd(tmp1, curDist0);
                _mm256_store_pd(tmp2, curDist0);
                _mm256_store_pd(tmp3, curDist0);
                for (int j = 0; j < 4; j++) {
                    #pragma omp atomic
                    dist[i * k + c] += tmp0[j];
                    #pragma omp atomic
                    dist[(i + 1) * k + c] += tmp1[j];
                    #pragma omp atomic
                    dist[(i + 2) * k + c] += tmp2[j];
                    #pragma omp atomic
                    dist[(i + 3) * k + c] += tmp3[j];
                }
            }
        }
    }
}

/**
 * @brief Selecting Nearest Centroid.
 *
 * @description
 * After having n * k distance matrix, we want to first find the minimum
 * distance to centroids for each point.
 * This would be a n dimensional vector, then we broadcast this vector to match
 * back to the n * k distance matrix to generate.
 * A n * k mask of 0 or 1 representing if this this the nearest centroid for
 * corresponding point.
 * This n * k mask could also be viewed as n k-dimensional one-hot vectors.
 *
 * @param n number of points
 * @param k number of centroids
 * @param dist n * k distance matrix (double)
 * @param mask n * k mask
 */
void kernel2(int n, int k, double* dist, double* mask) {
    for (int i = 0; i < n; i += 8) {
        __m256d a = _mm256_set_pd(DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
        __m256d b = _mm256_set_pd(DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
        
        for (int j = 0; j < k; j += 4) {
            // load
            __m256d a0, a1, a2, a3, b0, b1, b2, b3;
            a0 = _mm256_load_pd(&dist[i * k + j]);
            a1 = _mm256_load_pd(&dist[(i + 1) * k + j]);
            a2 = _mm256_load_pd(&dist[(i + 2) * k + j]);
            a3 = _mm256_load_pd(&dist[(i + 3) * k + j]);
            b0 = _mm256_load_pd(&dist[(i + 4) * k + j]);
            b1 = _mm256_load_pd(&dist[(i + 5) * k + j]);
            b2 = _mm256_load_pd(&dist[(i + 6) * k + j]);
            b3 = _mm256_load_pd(&dist[(i + 7) * k + j]);

            // transpose
            transpose16(&a0, &a1, &a2, &a3);
            transpose16(&b0, &b1, &b2, &b3);

            // find min
            a = _mm256_min_pd(a, a0);
            a = _mm256_min_pd(a, a1);
            a = _mm256_min_pd(a, a2);
            a = _mm256_min_pd(a, a3);
            b = _mm256_min_pd(b, b0);
            b = _mm256_min_pd(b, b1);
            b = _mm256_min_pd(b, b2);
            b = _mm256_min_pd(b, b3);

            // compare and mask
            a0 = _mm256_cmp_pd(a, a0, 0);
            a1 = _mm256_cmp_pd(a, a1, 0);
            a2 = _mm256_cmp_pd(a, a2, 0);
            a3 = _mm256_cmp_pd(a, a3, 0);
            b0 = _mm256_cmp_pd(b, b0, 0);
            b1 = _mm256_cmp_pd(b, b1, 0);
            b2 = _mm256_cmp_pd(b, b2, 0);
            b3 = _mm256_cmp_pd(b, b3, 0);

            // transpose back
            transpose16(&a0, &a1, &a2, &a3);
            transpose16(&b0, &b1, &b2, &b3);

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

/**
 * @brief Recalculating centroids.
 *
 * @description
 * Goal: Given a n * -k mask and a n * d points matrix, we want to re-calculate
 * the value of k centroids (a k * d matrix).
 * Process:
 * 1. Calculate total number of points inside each cluster and store in a
 * size k vector.
 * 2. Multiply n * k(k * n) mask with n * dim points. It can be implemented by
 * outer product. result: k * dim
 * 3. For each centoid each dimension, divide by the number of points of the
 * centroid (<= n). result: k * dim.
 *
 * @param n number of points
 * @param k number of centroids
 * @param dim number of dimensions
 * @param mask n * k mask (0 or 1, double).
 * @param points n * k mask (0 or 1, double).
 * @param centroids k * dim, k centroids each d dimensions (double).
 *
 */
void kernel3(int n,
             int k,
             int dim,
             double* mask,
             double* points,
             double* centroids) {
    // A vector k doubles.
    double* countGroup;
    posix_memalign((void**)&countGroup, 64, k * sizeof(double));
    // temporary content.
    __m256d tmp;
    // A patial output (4 x 4 matrix)
    __m256d output[4];
    // A point's 4 dimensions.
    __m256d currPoint;
    // Each maskx stores 4 same doubles.
    __m256d mask0, mask1, mask2, mask3;

    // Transform -naf into 1.0
    maskTransform(n, k, mask);

    // Calculate point number in each clusters
    for (int c = 0; c < k; c += 4) {
        tmp = _mm256_setzero_pd();
        for (int i = 0; i < n; i++) {
            tmp = _mm256_add_pd(_mm256_load_pd(&mask[i * k + c]), tmp);
        }
        _mm256_store_pd(&countGroup[c], tmp);
    }

#pragma omp parallel for num_threads(8)
    for (int c = 0; c < k; c += 4) {
        // Broadcast point number in 4 different clusters.
        __m256d count0 = _mm256_broadcast_sd(&countGroup[c]);
        __m256d count1 = _mm256_broadcast_sd(&countGroup[c + 1]);
        __m256d count2 = _mm256_broadcast_sd(&countGroup[c + 2]);
        __m256d count3 = _mm256_broadcast_sd(&countGroup[c + 3]);

        for (int d = 0; d < dim; d += 4) {
            // Initialize output data.
            output[0] = _mm256_setzero_pd();
            output[1] = _mm256_setzero_pd();
            output[2] = _mm256_setzero_pd();
            output[3] = _mm256_setzero_pd();

            for (int i = 0; i < n; i++) {
                // Load data from mask and points.
                currPoint = _mm256_load_pd(&points[i * dim + d]);
                mask0 = _mm256_broadcast_sd(&mask[i * k + c]);
                mask1 = _mm256_broadcast_sd(&mask[i * k + c + 1]);
                mask2 = _mm256_broadcast_sd(&mask[i * k + c + 2]);
                mask3 = _mm256_broadcast_sd(&mask[i * k + c + 3]);

                // Perform outer product, and add to output.
                output[0] = _mm256_fmadd_pd(mask0, currPoint, output[0]);
                output[1] = _mm256_fmadd_pd(mask1, currPoint, output[1]);
                output[2] = _mm256_fmadd_pd(mask2, currPoint, output[2]);
                output[3] = _mm256_fmadd_pd(mask3, currPoint, output[3]);
            }

            // Output divide by the number of points in each cluster.
            if (countGroup[c])
                output[0] = _mm256_div_pd(output[0], count0);
            if (countGroup[c + 1])
                output[1] = _mm256_div_pd(output[1], count1);
            if (countGroup[c + 2])
                output[2] = _mm256_div_pd(output[2], count2);
            if (countGroup[c + 3])
                output[3] = _mm256_div_pd(output[3], count3);

            // Store back
            _mm256_store_pd(&centroids[c * dim + d], output[0]);
            _mm256_store_pd(&centroids[(c + 1) * dim + d], output[1]);
            _mm256_store_pd(&centroids[(c + 2) * dim + d], output[2]);
            _mm256_store_pd(&centroids[(c + 3) * dim + d], output[3]);
        }
    }
}

/**
 * @brief Print four doubles saved in a _m256d variable.
 *
 * @param var a _m256d variable equals to an array of 4 doubles.
 */
void print256d(__m256d var) {
    double val[4];
    memcpy(val, &var, sizeof(val));
    printf("%.2lf %.2lf %.2lf %.2lf \n", val[0], val[1], val[2], val[3]);
}

/**
 * @brief Transpose a 4 x 4 matrix in place.
 *
 * @param A0 First row of matrix.
 * @param A1 Seconde row.
 * @param A2 Third row.
 * @param A3 Forth row.
 */
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

/**
 * @brief Transform '-naf' in mask into 1.0
 *
 * @param n point number
 * @param k cluster number
 * @param mask n * k mask
 */
void maskTransform(int n, int k, double* mask) {
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < k; c++) {
            mask[i * k + c] = !mask[i * k + c] ? 0.0 : 1.0;
        }
    }
}

int main(int argc, char** argv) {
    int N = 16, dim = 16, K = 4;
    if (argc != 2) {
        printf("kernel3.x <<N: Number points>> \n");
        exit(0);
    } else {
        N = atoi(argv[1]);
    }
    assert(N % 4 == 0);
    printf("N = %d, D = %d\n", N, dim);

    double *dist, *mask, *points, *centroids;
    unsigned long long t0 = 0, t1 = 0;

    for (K = 4; K <= 128; K *= 2) {
        posix_memalign((void**)&dist, 64, N * K * sizeof(double));
        posix_memalign((void**)&mask, 64, N * K * sizeof(double));
        posix_memalign((void**)&points, 64, N * dim * sizeof(double));
        posix_memalign((void**)&centroids, 64, K * dim * sizeof(double));

        for (int i = 0; i != N * K; ++i) {
            dist[i] = 0;
        }
        srand((unsigned)time(NULL));
        for (int i = 0; i < N * dim; i++) {
            points[i] = ((double)rand()) / ((double)RAND_MAX);
        }
        for (int i = 0; i < K * dim; i++) {
            centroids[i] = ((double)rand()) / ((double)RAND_MAX);
        }

        // printf("Points: \n");
        // for (int i = 0; i < N; i++) {
        //     for (int j = 0; j < dim; j++)
        //         printf("%.2f ", points[i * dim + j]);
        //     printf("\n");
        // }
        // printf("\n");

        // printf("Old centroid: \n");
        // for (int i = 0; i < K; i++) {
        //     for (int j = 0; j < dim; j++)
        //         printf("%.2lf ", centroids[i * dim + j]);
        //     printf("\n");
        // }
        // printf("\n");

        unsigned long long time1 = 0, time2 = 0;

        t0 = rdtsc();
        kernel1(points, centroids, dist, N, dim, K);
        t1 = rdtsc();
        time1 += (t1 - t0);

        // printf("Distance Matrix: \n");
        // for (int i = 0; i < N; i++) {
        //     for (int j = 0; j < K; j++)
        //         printf("%.2f ", dist[i * K + j]);
        //     printf("\n");
        // }
        // printf("\n");

        t0 = rdtsc();
        kernel2(N, K, dist, mask);
        t1 = rdtsc();
        time1 += (t1 - t0);

        // printf("Generated Mask: \n");
        // for (int i = 0; i < N; i++) {
        //     for (int j = 0; j < K; j++) {
        //         printf("%d ", mask[i * K + j] == 0.0 ? 0 : 1);
        //     }
        //     printf("\n");
        // }
        // printf("\n");
        t0 = rdtsc();
        kernel3(N, K, dim, mask, points, centroids);
        t1 = rdtsc();
        time2 += (t1 - t0);

        // printf("New centroid values: \n");
        // for (int i = 0; i < K; i++) {
        //     for (int j = 0; j < dim; j++)
        //         printf("%.2lf ", centroids[i * dim + j]);
        //     printf("\n");
        //

        double perf1 = (double)(2 * N * dim + N * K) / time1;
        double perf2 = (double)(N * dim) / time2;
        printf("K = %d, perf1 = %lf, perf2 = %lf\n", K, perf1, perf2);

        free(dist);
        free(mask);
        free(points);
        free(centroids);
    }

    return 0;
}
