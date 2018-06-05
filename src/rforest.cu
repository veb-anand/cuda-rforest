/* 
 * RandomForest
 * @author Vaibhav Anand, 2018
 */

#include <cstdio>
#include <cuda_runtime.h>

#include "rforest.cuh"
#include "cuda_header.cuh"
#include "utils.hpp"

/* For every point in gpu_in_x, with shape (arg:num_features, arg:num_points), 
calculate the impurity based on splitting the data by that point. */
 __global__
 void cuda_get_losses(const float *gpu_in_x, const float *gpu_in_y, 
        float *gpu_out_x, int num_features, int num_points) {
    extern __shared__ float shmem[];

    unsigned tid = threadIdx.x; // current index in warp (maximum 32)
    float part1_y;  // sum of y where col >= val (partition 1)
    float part1_n;  // number of rows where col >= val
    float part2_y;  // sum of y where col < val (partition 2)
    float part2_n;  // number of rows where col < val

    /* For every value in "data", compute impurity by splitting at that value. 
    Serially do this for every point p. */
    for (uint p = 0; p < num_points; p++) {
        /* Impurity is calculated by performing operations to every point in 
        gpu_in_x. Each block takes a different feature in gpu_in_x and i is the 
        offset to this feature. ip is the index of a point in the feature, 
        within [0, num_points). */
        uint i = (blockIdx.x * num_points);
        uint ip = tid; 

        /* For every point p, each block calculates the impurity for every 
        feature, at value (i + p). */
        uint j = i + p; // match to first point in every feature of gpu_in

        /* Since num_points can be greater than THREADS_PER_BLOCK, we iterate 
        through gpu_in_x, storing results serially into shared memory until we 
        are done. */

        /* This part is not in the while loop below because we want to set, 
        not add in the first iteration (b/c of repeated kernel calls) and this 
        saves time by not having to execute statements to set it to 0 outside 
        the while loop. */
        if (ip < num_points) {
            /* part1_n before summing for all points. */
            shmem[4 * tid] = (gpu_in_x[i + ip] >= gpu_in_x[j]);
            
            /* part1_y before summing for all points. */
            shmem[4 * tid + 1] = shmem[4 * tid] * gpu_in_y[tid];

            /* part2_y before summing for all points. */
            shmem[4 * tid + 2] = (1. - shmem[4 * tid]) * gpu_in_y[tid];
            ip += THREADS_PER_BLOCK;
        }
        while (ip < num_points) {
            shmem[4 * tid] += (gpu_in_x[i + ip] >= gpu_in_x[j]);
            shmem[4 * tid + 1] += shmem[4 * tid] * gpu_in_y[tid];
            shmem[4 * tid + 2] += (1 - shmem[4 * tid]) * gpu_in_y[tid];
            ip += THREADS_PER_BLOCK;
        }
        __syncthreads();

        /* Since we have allocated 4 times as many values as there are threads 
        in each block, sum the last quarter of values into the first quarter 
        manually before reducing over the first three quarters, which  
        (tid + blockDim.x * 2) covers. */
        shmem[tid] += shmem[tid + blockDim.x * 3];

        /* We only reduce until s>2 such that the first four values of shared 
        memory correspond to part1_n, part1_y, part2_y, and NA. */
        for (uint s = blockDim.x * 2; s > 2; s >>= 1) {
            __syncthreads();
            if (tid < s) {
                shmem[tid] += shmem[tid + s];
                shmem[tid + s] = 0.; // for subsequent kernel runs
            }
        }

        __syncthreads();

        /* After aggregating for every point, each block uses one thread to 
        compute the impurity for its feature. */
        if (tid == 0) {
            part1_n = shmem[0];

            /* All points not in partition 1 are in partition 2 of the data. */
            part2_n = num_points - part1_n;

            /* part1_y becomes fraction of y=1 where col >= val */
            part1_y = shmem[1] / (part1_n + SMALL);

            /* part2_y becomes fraction of y=1 where col < val */
            part2_y = shmem[2] / (part2_n + SMALL);

            /* Get proportion of points that are in partition 1 vs 2. */
            float part1_p = part1_n / num_points;
            
            /* Store total impurity by splitting into partitions 1 and 2. */
            gpu_out_x[blockIdx.x * num_points + p] = (GINI(part1_y) * part1_p 
                + GINI(part2_y) * (1 - part1_p));
        }
    }
}


/* Call the cuda_get_losses kernel. This can be exposed to non-CUDA files. Note 
that arg:num_features does not include y as a feature, unlike most other 
functons. Also, please read section "hardware assumptions" in README.txt. */
void cuda_call_get_losses(float *gpu_in_x, float *gpu_in_y, float *gpu_out_x,
    int num_features, int num_points) {
    cuda_get_losses<<<num_features, THREADS_PER_BLOCK, 
        (4 * THREADS_PER_BLOCK * sizeof(float))>>>(
        gpu_in_x, gpu_in_y, gpu_out_x, num_features, num_points);
}
