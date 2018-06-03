/* 
 * Random Forest
 * Vaibhav Anand, 20148
 */

#ifndef RFOREST_DEVICE_CUH
#define RFOREST_DEVICE_CUH

#include "cuda_header.cuh"

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <iostream>


void cuda_call_mat_gt_y(float *gpu_in_x, float *gpu_in_y, float *gpu_tmp,
    float *gpu_out_x, int size_x, int num_points);


#endif

