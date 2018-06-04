/* 
 * Random Forest
 * Vaibhav Anand, 2018
 */


#ifndef RFOREST_DEVICE_CUH
#define RFOREST_DEVICE_CUH

#include "cuda_header.cuh"

#pragma once


void cuda_call_get_losses(float *gpu_in_x, float *gpu_in_y, float *gpu_out_x, 
    int num_features, int num_points);


#endif
