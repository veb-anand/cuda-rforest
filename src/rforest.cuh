/* 
 * RandomForest
 * @author Vaibhav Anand, 2018
 */

#pragma once

#define THREADS_PER_BLOCK 1024

/* Call a kernel, which, for every point in gpu_in_x, with shape 
(arg:num_features, arg:num_points), calculates the impurity based on splitting 
the data by that point. */
void cuda_call_get_losses(float *gpu_in_x, float *gpu_in_y, float *gpu_out_x, 
    int num_features, int num_points);

