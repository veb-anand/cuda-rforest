/* 
 * Random Forest
 * Vaibhav Anand, 2018
 */

#pragma once

#include <cublas_v2.h>
#include "utils.hpp"

/**
 * TODO: comment
 */
class RandomForest {
public:
    RandomForest(int num_trees, int max_depth, float sub_sampling, 
        bool gpu_mode);
    ~RandomForest();
    
    void fit(float *data, int num_features, int num_points);
    float *predict(float *x, int num_points);

    // these functions are relatively generic but are included because they contain personalized cuda versions
    float get_mse_loss(float *y, float *preds, int num_points);
    float *transpose(float *data, int num_rows, int num_cols);
    void start_time();
    float end_time();
    void print_forest(int verbosity);

private:
    float get_info_loss(float *y, float *col, float val, int num_points);
    node *data_split(float *data, int num_points, bool no_split);
    node *node_split(float *data, int num_points, int depth);
    void build_forest(int sub_sample_size);
    float predict_point(float *point, node *n);

    float *data;
    int num_features; // includes y
    int num_points;

    bool gpu_mode;

    int num_trees;
    node **forest;
    float sub_sampling;
    int max_depth;

    cublasHandle_t cublasHandle;
    float *gpu_in_x, *gpu_in_y, *gpu_out_x;
    cudaEvent_t start, stop;
};
