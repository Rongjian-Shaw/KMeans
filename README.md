# KMeans

## How to run the code

1. The server we use is ECE000.

2. unzip my-kmeans.zip

3. cd my-kmeans

4. chmod 766(if you have not permission to use ./compile.sh)

5. ./compile.sh

6. ./kernels.x 1024  (You can change the 1024 to other number, it represents the size of data)

   



## Future direction

### Cache

In the current code, we do not use cache optimization. If we have more time, we will research the cache size(how many sets and ways the server is) of the current server, and then use the **blocking** method to optimize the project and make the code faster.



### Parallelization

Currently, We are using **#pragma omp atomic** to avoid the confliction of parallelization. However, **#pragma omp reduction** actually can get the same result, but we met some bugs here so we not use the reduction. If we can use reduction, we believe that we will get faster speed.



###



## Abstract

In this project, we optimize the KMeans algorithm and make it faster. 

