/* 
 * RandomForest
 * @author Vaibhav Anand, 2018
 */

#include <cstdlib>
#include <string.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "rforest.hpp"
#include "helper_cuda.h"
#include "utils.hpp"
#include "rforest.cuh"
#include "constants.hpp"

using namespace std;

/* The constructor takes in arguments that parameterizes the model. */
RandomForest::RandomForest(int num_trees, int max_depth, float sub_sampling, 
        bool gpu_mode) {
    /* Nullify these variables in case the user attempts to call a method 
    that uses before they are set. */
    this->data = NULL;
    this->forest = NULL;
    this->num_features = -1;
    this->num_points = -1;
    this->gpu_in_x = NULL;
    this->gpu_in_y = NULL;
    this->gpu_out_x = NULL;

    /* Set these model parameters based on constructor's arguments. */
    this->num_trees = num_trees;
    this->max_depth = max_depth;
    this->sub_sampling = sub_sampling;
    this->gpu_mode = gpu_mode;
}

/* The destructor frees GPU resources is allocated and components of the 
random forest if fit was called. */
RandomForest::~RandomForest() {
    /* Free all arrays and handles allocated and created for CUDA/CUBLAS. */
    if (this->gpu_in_x != NULL) {
        CUDA_CALL(cudaFree(this->gpu_in_x));
        CUDA_CALL(cudaFree(this->gpu_in_y));
        CUDA_CALL(cudaFree(this->gpu_out_x));
        CUBLAS_CALL(cublasDestroy(this->cublasHandle));    
    }
    
    /* Free every tree of nodes in random forest. */
    if (this->forest != NULL) {
        for (int t = 0; t < this->num_trees; t++) 
            free_tree(forest[t]);

        free(this->forest);
    }
}

/* Fits the model by building a random forest trained on arg:data, described by
num_features and num_points.*/
void RandomForest::fit(float *data, int num_features, int num_points) {
    /* Set these model parameters based on constructor's arguments. */
    this->data = data;
    this->num_features = num_features;
    this->num_points = num_points;

    /* If using the GPU, allocate the necessary arrays. */
    if (gpu_mode && (this->gpu_in_x == NULL)) {
        CUBLAS_CALL(cublasCreate(&cublasHandle));

        /* Allocate arrays/matrices on gpu. Allocate with the largest size that they will be used for. */
        int size_x = (num_features - 1) * num_points;
        CUDA_CALL(cudaMalloc((void **) &this->gpu_in_x, 
            size_x * sizeof(float)));
        CUDA_CALL(cudaMalloc((void **) &this->gpu_in_y, 
            num_points * sizeof(float)));
        CUDA_CALL(cudaMalloc((void **) &this->gpu_out_x, 
            size_x * sizeof(float)));
    }

    /* Now construct the forest. */
    this->build_forest((int) (this->sub_sampling * num_points));
}

/* Predicts and returns vector for input matrix arg:test_x with arg:num_points.
Takes data without y vector in point-major order (feature-major is used
everywhere else). Also, the number of features in test_x should remain
num_features used in training.*/
float *RandomForest::predict(float *test_x, int num_points) {
    if (this->forest == NULL) {
        printf("Invalid call to predict(). Cannot predict unless model has "
            "been trained.\n");
    }

    /* Allocated array for predictions. */
    float *preds = (float *) malloc(num_points * sizeof(float));
    if (preds == NULL)  malloc_err("predict");

    /* Caculate predicted class for each point in test_x serially. */
    for (int p = 0; p < num_points; p++) {
        preds[p] = 0.;
        for (int t = 0; t < num_trees; t++) {
            preds[p] += predict_point(test_x + p * (this->num_features - 1), 
                this->forest[t]);
        }
        preds[p] = (preds[p] * 2) > num_trees; // get mode
    }
    
    return preds;
}

/* Returns the loss, as mean-squared-error, given vectors arg:y and arg:preds to
compare, each with arg:num_points.*/
float RandomForest::get_mse_loss(float *y, float *preds, int num_points) {
    float loss;

    /* If gpu mode and we have enough room in pre-allocated gpu_in_y and 
    gpu_out_x, then use cublas to calculate error. */
if (this->gpu_mode && (num_points <= this->num_points)) {
    float negone = -1.0;
    
    /* Copy y into gpu_in_y and preds into gpu_out_x. On small datasets, this 
    step makes using the GPU inefficient. */
    CUDA_CALL(cudaMemcpy(this->gpu_in_y, y, num_points * sizeof(float), 
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(this->gpu_out_x, preds, num_points * sizeof(float), 
        cudaMemcpyHostToDevice));
    
    /* preds = preds - y */
    CUBLAS_CALL(cublasSaxpy(this->cublasHandle, num_points, &negone, 
        this->gpu_out_x, 1, this->gpu_in_y, 1));
    /* loss = sum(preds^2) = sum(abs(preds)) [since preds is 0 or 1] */
    CUBLAS_CALL(cublasSnrm2(this->cublasHandle, num_points, this->gpu_in_y, 
        1, &loss));
    loss *= loss; // because Snrm2 returns sqrt(sum(squares))
} 
else {
    if (this->gpu_mode) {
        printf("Warning: Could not use gpu for get_mse_loss()\n");
    }

    /* Compute loss = sum(abs(y - preds)). */
    loss = 0.;
    for (int i = 0; i < num_points; i++) {
        loss += fabs(y[i] - preds[i]);
    }
}

    loss /= num_points; // loss is 'mean' squared error.
    return loss;
}

/* Returns the transpose (not in-place) of a matrix arg:data with arg:num_rows 
and arg:num_cols. It is generic but done differently based on this->gpu_mode 
(if true, it uses CUBLAS).*/
float *RandomForest::transpose(float *data, int num_rows, int num_cols) {
    /* Allocate array on host that will hold transpose of arg:data. */
    int size = num_rows * num_cols;
    float *t = (float *) malloc(size * sizeof(float));
    if (t == NULL) malloc_err("transpose");

    /* If gpu mode and we have enough room in pre-allocated gpu_in_x and 
    gpu_out_x, then use cublas to calculate transpose. */
if ((this->gpu_mode) && ((this->num_features - 1) * this->num_points >= size)) {
    float one = 1., zero = 0.;
    CUDA_CALL(cudaMemcpy(this->gpu_in_x, data, size * sizeof(float), 
        cudaMemcpyHostToDevice));
    // use cublasSgeam() w/alpha=1 and beta=0
    CUBLAS_CALL(cublasSgeam(this->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
        num_rows, num_cols, &one, this->gpu_in_x, num_cols, &zero, 
        this->gpu_in_x, num_rows, this->gpu_out_x, num_rows));
    CUDA_CALL(cudaMemcpy(t, this->gpu_out_x, size * sizeof(float), 
        cudaMemcpyDeviceToHost));
}
else {
    if (this->gpu_mode) {
        printf("Warning: Could not use gpu for transpose\n");
    }

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            t[j * num_rows + i] = data[i * num_cols + j];
        }
    }
}

    return t;
}

/* Start timing using CUDA. */
void RandomForest::start_time() {
    CUDA_CALL(cudaEventCreate(&this->start));
    CUDA_CALL(cudaEventCreate(&this->stop));
    CUDA_CALL(cudaEventRecord(this->start));
}

/* Stop timing and return the elapsed time in milliseconds. */
float RandomForest::end_time() {
    float time = -1.;
    CUDA_CALL(cudaEventRecord(this->stop));
    CUDA_CALL(cudaEventSynchronize(this->stop));
    CUDA_CALL(cudaEventElapsedTime(&time, this->start, this->stop));
    CUDA_CALL(cudaEventDestroy(this->start));
    CUDA_CALL(cudaEventDestroy(this->stop));
    return time;
}

/* Prints verbosity number of trees (if they exist), in the format described in
constants.hpp. */
void RandomForest::print_forest(int num_print) {
    if (this->forest == NULL) {
        printf("Invalid call to print_forest(). Cannot print models unless "
            "they have been trained.\n");
    }

    for (int t = 0; t < MIN(this->num_trees, num_print); t++) {
        print_tree(this->forest[t]); 
        printf("\n\n");
    }
}

/* Called by fit(). Serially trains each decision tree in the random forest.
If sub_sample_size > 0, train each tree on a subsample of the data. */
void RandomForest::build_forest(int sub_sample_size) {
    this->forest = (node **) malloc(this->num_trees * sizeof(node *));
    if (this->forest == NULL) malloc_err("build_forest:0");

    /* Allocate space for subsamples of the training data. */
    float *sub = NULL;
    if (sub_sample_size > 0.) {
        sub = (float *) malloc(sub_sample_size * this->num_features * 
            sizeof(float));
        if (sub == NULL) malloc_err("build_forest:1");
    }

    for (int t = 0; t < this->num_trees; t++) {
        /* Create subsample of data if necessary. Then train tree on it. */
        if (sub != NULL) {
            for (int s = 0; s < sub_sample_size; s++) {
                int p = rand() % num_points;
                for (int f = 0; f < this->num_features; f++)
                    sub[f * sub_sample_size + s] = this->data[f 
                        * num_points + p];
            }
            this->forest[t] = this->node_split(sub, sub_sample_size, 0);
        } else {
            this->forest[t] = this->node_split(this->data, this->num_points, 0);
        }
    }
    
    if (sub != NULL) free(sub);
}

/* Recursively predicts class of arg:point (vector) using tree node:n. */
float RandomForest::predict_point(float *point, node *n) {
    if (n->feature == 0)
        return n->val;
    
    if (point[n->feature - 1] >= n->val)
        return predict_point(point, n->true_branch);
    else
        return predict_point(point, n->false_branch);
}

/* Recursively build and return a decision tree by splitting arg:data into two 
branches, until a depth limit is reached or no more information can be gain by 
splitting.*/
node *RandomForest::node_split(float *data, int num_points, int depth) {
    /* Find the optimal split in the data. */
    node *n = this->data_split(data, num_points, (depth >= this->max_depth));

    /* Split did not help increase information, so stop. */
    if (n->feature == 0) {
        return n;
    }

    /* Partition data according to the split. */
    int t_points = 0, f_points, tp = 0, fp = 0;
    float *col = data + (num_points * n->feature);

    /* Compute number of rows in each partition. */
    for (int p = 0; p < num_points; p++) t_points += (col[p] >= n->val);
    f_points = num_points - t_points;

    /* Allocate matrices to hold true/false partitions of data. */
    float *t_data = (float *) malloc(t_points * 
        this->num_features * sizeof(float));
    float *f_data = (float *) malloc(f_points * 
        this->num_features * sizeof(float));
    if ((t_data == NULL) || (f_data == NULL)) malloc_err("node_split");

    /* Poor indexing, but faster than transposing back and forth. */
    for (int p = 0; p < num_points; p++) {
        if (col[p] >= n->val) {
            for (int f = 0; f < this->num_features; f++) {
                t_data[(t_points * f) + tp] = data[(num_points * f) + p];
            }
            tp++;
        } else {
            for (int f = 0; f < this->num_features; f++) {
                f_data[(f_points * f) + fp] = data[(num_points * f) + p];
            }
            fp++;
        }
    }
    
    /* Recursively build tree on true and false partitions. */
    n->false_branch = this->node_split(f_data, f_points, depth + 1);
    free(f_data);
    n->true_branch = this->node_split(t_data, t_points, depth + 1);
    free(t_data);

    return n;
}

/* Finds and returns optimal split in the data (by attribute and by val) that
 minimizes impurity. If arg:no_split, the tree has reached its maximum depth and
 the split will contain be defined as such and return before finding the optimal
 split.*/
node *RandomForest::data_split(float *data, int num_points, bool no_split) {
    node *n = new node;
    float info_gain;

    /* Get unc (uncertainty/impurity) in all the data (note that y is the 
    first arg:num_points of arg:data). */
    float y_true = 0.; // number of true y in this partition
    for (int p = 0; p < num_points; p++) {
        y_true += data[p];
    }

    /* If there is no impurity, then we cannot gain info, so return. */
    if (no_split || (y_true < 1.) || (y_true > num_points - 1)) {
        n->feature = 0;
        n->val = (y_true * 2) > num_points;
        return n;
    }
    
    float unc = GINI(y_true / num_points);

    int f = this->num_features - 1;
    int size_x = f * num_points;

    /* Use GPU if this->gpu_mode and there are enough points to make it 
    efficient (based on num_points and num_features). */
if(this->gpu_mode && (size_x > GPU_BARRIER_1) && (num_points > GPU_BARRIER_2)) {
    /* Copy data into inputs. */
    CUDA_CALL(cudaMemcpy(this->gpu_in_x, data + num_points, 
        size_x * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(this->gpu_in_y, data, num_points * sizeof(float), 
        cudaMemcpyHostToDevice));

    /* Compute information impurities. */
    cuda_call_get_losses(this->gpu_in_x, this->gpu_in_y, this->gpu_out_x, f, 
        num_points);
    int result;

    /* Get the optimal impurity index */
    CUBLAS_CALL(cublasIsamin(this->cublasHandle, size_x, this->gpu_out_x, 1, 
        &result));
    result -= 1; // CUDA uses 1-indexing

    /* Get the optimal impurity at index and save it in the split n. */
    CUDA_CALL(cudaMemcpy(&info_gain, this->gpu_out_x + result, sizeof(float), 
        cudaMemcpyDeviceToHost));

    /* Save optimal split into n. */
    result += num_points;
    n->feature = (result / num_points);
    n->val = data[result];
} 
else {
    /* Now, determine what split causes the smallest information loss. */
    info_gain = unc;
    float info_impurity;

    /* For every possible split (using every point in data), find the 
    information loss using get_info_loss(). */
    for (int i = num_points; i < (size_x + num_points); i += num_points) {
        for (int p = 0; p < num_points; p++) {
            info_impurity = this->get_info_loss(data, data + i, 
                data[i + p], num_points);

            if (info_impurity < info_gain) {
                info_gain = info_impurity;
                n->feature = i / num_points;
                n->val = data[i + p];
            }
        }
    }
}

    /* Definition of information gain. Until now, info_gain was impurity. */
    info_gain = unc - info_gain;

    /* Set the partition to be the mode of y, since no possible info gain. */
    if (info_gain <= SMALL) {
        n->feature = 0;
        n->val = (y_true * 2) > num_points;
    }

    return n;
}

/* Get the information lost (impurity in y) by splitting data across feature by 
val. y and feature are of length num_points. This is cpu-only. */
float RandomForest::get_info_loss(float *y, float *feature, float val, 
        int num_points) {
    float part1_y = 0.;  // sum of y where col >= val (partition 1)
    float part1_n = SMALL;  // number of rows where col >= val
    float part2_y = 0.;  // sum of y where col < val (partition 2)
    float part2_n = SMALL;  // number of rows where col < val

    for (int p = 0; p < num_points; p++) {
        if (feature[p] >= val) {
            part1_y += y[p];
            part1_n += 1.;
        } else {
            part2_y += y[p];
            part2_n += 1.;
        }
    }
    part1_y /= part1_n; // fraction of y=1 where col >= val
    part2_y /= part2_n; // fraction of y=1 where col < val

    /* Get proportion of points that are in partition 1 vs 2. */
    float part1_p = (part1_n / num_points);

    /* Return total impurity by spliting into partitions 1 and 2. */
    return (GINI(part1_y) * part1_p + GINI(part2_y) * (1 - part1_p));
}
