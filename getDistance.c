/*
Calculate the distance between every centroid and data points

n: number of points
k: number of centroids
dist: n * k distance matrix (double)

1. Traverse every centroid
2. Get the first 4 elements of the first centroid
3. Traverse every data point
4. Get the first 4 elements of every point
5. Use the first 4 elements of a centroid and the first 4 elements of every point to calculate
    the first 4 dimensions distance and add it to the dist
6. Get the second 4 elements, repeat 2~5, until finishing all dimensions of the first centroid
7. Go to the next centroid, do 2~6, until finishing all of centroids. Then we get dist[n][k].
    dist[j][i] means the distance between the j th data point and i th centroid

*/


double** getDistance(double* data, double* centroids, int n, int d, int k)
{
    int kernelWidth = 4;
    int kernelLength = 4;
    double* dist;
    posix_memalign((void**) &dist, 64, n * k * sizeof(double));

    for(int i=0; i < k; i++) {
        for(int kernelHead=0; kernelWidth < d; kernelHead += 4){
            SIMDC = _mm256_load_pd(&centroids[i*d + kernelHead]); // [C[0][0], C[0][1], C[0][2], C[0][3]]    C[centorids][d]
            for(int j=0; j < n; j++) {
                _m256d curDist = _mm256_load_pd(&dist[j*k + i]);
                _m256d SIMDX = _mm256_load_pd(&data[j*d + kernelHead]); // [X[0][0], X[0][1], X[0][2], X[0][3]]    X[data][d]
                _m256d curDiff = _mm256_sub_pd(SIMD); // X - C
                curDist = _mm256_fmadd_pd(curDiff, curDiff, curDist); // (X-C)^2
                _mm256_store_pd(&dist[j*k + i], curDist)
            }            
        }
        
    }
}