/* 
 * RandomForest
 * @author Vaibhav Anand, 2018
 */

#pragma once

#include <cublas_v2.h>
#include "utils.hpp"

/**
 * RandomForest class. The constructor takes in arguments that parameterizes 
 * the model which can then be used to fit and predict on data and measure loss, 
 * while timing operations and inspecting the forest.
 */
class RandomForest {
public:
    RandomForest(int num_trees, int max_depth, float sub_sampling, 
        bool gpu_mode);
    ~RandomForest();
    
    /* Fits the model by building a random forest trained on arg:data, described
    by num_features and num_points.*/
    void fit(float *data, int num_features, int num_points);

    /* Predicts and returns vector for input matrix arg:test_x with 
    arg:num_points. */
    float *predict(float *test_x, int num_points);

    /* Returns the loss, as mean-squared-error, given vectors arg:y and 
    arg:preds to compare, each with arg:num_points. */
    float get_mse_loss(float *y, float *preds, int num_points);

    /* Returns the transpose of a matrix arg:data with arg:num_rows and 
    arg:num_cols. It is generic but done differently based on this->gpu_mode. */
    float *transpose(float *data, int num_rows, int num_cols);

    /* Use start_time() to start timing. Call end_time() to stop timing and
    return the elapsed time in milliseconds. Implemented using CUDA 
    event-timing functions.*/
    void start_time();
    float end_time();

    /* Prints verbosity number of trees (if they exist), in the format 
    described in constants.hpp. */
    void print_forest(int verbosity);

private:
    /* Called by fit(). Serially trains each decision tree in the random forest.
    If sub_sample_size > 0, train each tree on a subsample of the data. */
    void build_forest(int sub_sample_size);

    /* Recursively predicts class of arg:point (vector) using tree node:n. */
    float predict_point(float *point, node *n);

    /* Recursively build and return a decision tree by splitting arg:data into 
    two branches, until a depth limit is reached or no more information can 
    be gain by splitting. */
    node *node_split(float *data, int num_points, int depth);

    /* Finds and returns the optimal split of arg:data that results in the 
    highest  information gain. If arg:no_split, the tree has reached its 
    maximum depth and the split will contain be defined as such and return 
    before finding the optimal split. */
    node *data_split(float *data, int num_points, bool no_split);

    /* Computes and returns the information loss (impurity) by splitting 
    the dataset's feature arg:feature by arg:val. This is CPU-only. */
    float get_info_loss(float *y, float *feature, float val, int num_points);

    /* The dataset used to train the model. It is in feature-major order for 
    optimizing several training bottlenecks. */
    float *data;
    int num_features; // includes y
    int num_points;

    /* If true, the model will train using the GPU (CUDA and CUBLAS). */
    bool gpu_mode;

    /* Number of trees (num_trees) to train, each training to max_depth 
    using sub_sampling fraction of the dataset. */
    int num_trees;
    int max_depth;
    float sub_sampling; // if <= 0, no subsampling occurs
    
    /* An array of decision trees (as node *) that make up the random forest. */
    node **forest;

    /* These resources will be allocated and used to implement training on the
    GPU and event timing. Their contexts vary between functions. */
    cublasHandle_t cublasHandle;
    float *gpu_in_x, *gpu_in_y, *gpu_out_x;
    cudaEvent_t start, stop;
};
