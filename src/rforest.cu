/* 
 * Random Forest
 * Vaibhav Anand, 2018
 */

#include "rforest.cuh"

#include <cstdio>
#include <cuda_runtime.h>

#include "cuda_header.cuh"
#include "utils.hpp"

#define THREADS_PER_BLOCK 1024


 __global__
 void cuda_get_losses(const float *gpu_in_x, const float *gpu_in_y, 
        float *gpu_tmp, float *gpu_out_x, int num_features, int num_points) {
    extern __shared__ float shmem[];

    unsigned tid = threadIdx.x;
    float part1_n, part1_y, part2_n, part2_y;
    // uint threads_per_block = blockDim.x;

    if (tid > (num_points * 4 - 2)) shmem[tid] = 0.;
    __syncthreads();

    for (uint col = 0; col < num_points; col++) {
        uint k = (blockIdx.x * num_points) + tid;

        if (blockIdx.x < num_features) {
            // shmem[tid] = 0.;
            uint l = (blockIdx.x * num_points) + col;

            // ITERATE by THREADS_PER_BLOCK until we are done with num_points!

            if (tid < num_points) {
                gpu_tmp[k] = (gpu_in_x[k] >= gpu_in_x[l]);
                shmem[4 * tid] = gpu_tmp[k];
                shmem[4 * tid + 1] = shmem[4 * tid] * gpu_in_y[tid];
                shmem[4 * tid + 2] = (1 - shmem[4 * tid]) * gpu_in_y[tid];
                // atomicAdd(&shmem[0], gpu_tmp[k]);
            }
            
            // TODO: num_points better be even!
            for (uint s = blockDim.x / 2; s > 2; s >>= 1) {
                if (tid < s) {
                    shmem[tid] += shmem[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                part1_n = shmem[0];
                part2_n = num_points - part1_n;
                part1_y = shmem[1] / (part1_n + 0.00001);
                part2_y = shmem[2] / (part2_n + 0.00001);
                float part1_p = part1_n / num_points;
                float result = GINI(part1_y) * part1_p + GINI(part2_y) * (1 - part1_p);
                // purposely done w/blockIdx.x as row-indexer
                gpu_out_x[blockIdx.x * num_points + col] = result;
            }
        }   
    }
}



// num_features does not include y
void cuda_call_get_losses(float *gpu_in_x, float *gpu_in_y, float *gpu_tmp,
    float *gpu_out_x, int num_features, int num_points) {
    // TODO: constraint w/shared memory
    // change to non-hard-coded variables
    cuda_get_losses<<<num_features, THREADS_PER_BLOCK, 
        (4 * THREADS_PER_BLOCK * sizeof(float))>>>(
        gpu_in_x, gpu_in_y, gpu_tmp, gpu_out_x, num_features, num_points);
}

