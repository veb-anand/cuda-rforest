/* 
 * Random Forest
 * Vaibhav Anand, 20148
 */

#include "rforest.cuh"

#include <cstdio>
#include <cuda_runtime.h>

#include "cuda_header.cuh"

#define GINI(x) (1 - x * x - (1 - x) * (1 - x))
#define MIN(a, b) (a > b ? b : a)
#define MAX(a, b) (a > b ? a : b)

// CUDA_CALLABLE
// void cuda_blur_kernel_convolution(uint thread_index, const float* gpu_raw_data,
//                                   const float* gpu_blur_v, float* gpu_out_data,
//                                   const unsigned int n_frames,
//                                   const unsigned int blur_v_size) {
//     /* The gaussian convolution function that should be completed for each 
//     thread_index. */

//      Set blur_size_bounded = MIN(thread_index + 1, blur_v_size) 
//     int blur_size_bounded = (!((1 + thread_index) < blur_v_size) ? 
//         blur_v_size : (1 + thread_index));

//     /* For up to blur_v_size indices before thread_index that exist, set the 
//     output data to be the blur convoluted with the raw data. */
//     for (int j = 0; j < blur_size_bounded; j++)
//         gpu_out_data[thread_index] += gpu_raw_data[thread_index - j] * gpu_blur_v[j];
    
// }

// __global__
// void cuda_blur_kernel(const float *gpu_raw_data, const float *gpu_blur_v,
//                       float *gpu_out_data, int n_frames, int blur_v_size) {
//     /* Compute the current thread index. */
//     uint thread_index = blockDim.x * blockIdx.x + threadIdx.x;

//     /* Perform the convolution for all n_frames in the data. */
//     while (thread_index < n_frames) {
//          Do computation for this thread index. 
//         cuda_blur_kernel_convolution(thread_index, gpu_raw_data,
//                                      gpu_blur_v, gpu_out_data,
//                                      n_frames, blur_v_size);

//         /* Update the thread index to the next (blockDim.x * gridDim.x) threads
//         that will be done in parallel. */
//         thread_index += blockDim.x * gridDim.x;
//     }
// }

 __global__
 void cuda_mat_gt_y(const float *gpu_in_x, const float *gpu_in_y, 
        float *gpu_tmp, float *gpu_out_x, int size_x, int num_points) {
    extern __shared__ float shmem[];

    unsigned tid = threadIdx.x;
    float part1_n, part1_y, part2_n, part2_y;
    // uint threads_per_block = blockDim.x;

    if (tid > (num_points * 4 - 2)) shmem[tid] = 0.;
    __syncthreads();

    for (uint col = 0; col < num_points; col++) {
        uint k = (blockIdx.x * num_points) + tid;

        if (blockIdx.x < (size_x / num_points)) {
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


void cuda_call_mat_gt_y(float *gpu_in_x, float *gpu_in_y, float *gpu_tmp,
    float *gpu_out_x, int size_x, int num_points) {
    // TODO: constraint w/shared memory
    // change to non-hard-coded variables
    cuda_mat_gt_y<<<50, 1024, (4 * 1024 * sizeof(float))>>>(
        gpu_in_x, gpu_in_y, gpu_tmp, gpu_out_x, size_x, num_points);
}

// float cuda_call_blur_kernel(const unsigned int blocks,
//                             const unsigned int threads_per_block,
//                             const float *raw_data,
//                             const float *blur_v,
//                             float *out_data,
//                             const unsigned int n_frames,
//                             const unsigned int blur_v_size) {
//     // Use the CUDA machinery for recording time
//     cudaEvent_t start_gpu, stop_gpu;
//     float time_milli = -1;
//     cudaEventCreate(&start_gpu);
//     cudaEventCreate(&stop_gpu);
//     cudaEventRecord(start_gpu);

    
//     /* Allocate GPU memory for the raw input data. */
//     float* gpu_raw_data;
//     gpu_errchk(cudaMalloc((void **) &gpu_raw_data, n_frames * sizeof(float)));

//     /* Copy the data in raw_data into the GPU memory allocated. */
//     gpu_errchk(cudaMemcpy(gpu_raw_data, raw_data, n_frames * sizeof(float), cudaMemcpyHostToDevice));


//     /* Allocate GPU memory for the impulse signal. */
//     float* gpu_blur_v;
//     gpu_errchk(cudaMalloc((void **) &gpu_blur_v, blur_v_size * sizeof(float)));

//     /* Copy the data in blur_v into the GPU memory you allocated. */
//     gpu_errchk(cudaMemcpy(gpu_blur_v, blur_v, blur_v_size * sizeof(float), cudaMemcpyHostToDevice));


//      Allocate GPU memory to store the output audio signal after the 
//     convolution and set it to all zeros (convolution will assume zeros). 
//     float* gpu_out_data;
//     gpu_errchk(cudaMalloc((void **) &gpu_out_data, n_frames * sizeof(float)));
//     gpu_errchk(cudaMemset(gpu_out_data, 0., n_frames * sizeof(float)));
    
//     /* Call the blur kernel function. */
//     cuda_blur_kernel<<<blocks, threads_per_block>>>(gpu_raw_data, gpu_blur_v, 
//         gpu_out_data, n_frames, blur_v_size);

//     // Check for errors on kernel call
//     cudaError err = cudaGetLastError();
//     if (cudaSuccess != err)
//         fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
//     else
//         fprintf(stderr, "No kernel error detected\n");

//     /* Copy the output signal back from the GPU to host memory in out_data. */
//     gpu_errchk(cudaMemcpy(out_data, gpu_out_data, n_frames * sizeof(float), cudaMemcpyDeviceToHost));

//     /*Since the GPU computations are finished, free the GPU resources. */
//     gpu_errchk(cudaFree(gpu_raw_data));
//     gpu_errchk(cudaFree(gpu_blur_v));
//     gpu_errchk(cudaFree(gpu_out_data));

//     // Stop the recording timer and return the computation time
//     cudaEventRecord(stop_gpu);
//     cudaEventSynchronize(stop_gpu);
//     cudaEventElapsedTime(&time_milli, start_gpu, stop_gpu);
//     return time_milli;
// }
