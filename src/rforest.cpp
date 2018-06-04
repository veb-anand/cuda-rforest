/* 
 * Random Forest
 * Vaibhav Anand, 2018
 */

#include <cstdio>
#include <cstdlib>
// #include <cmath>
// #include <cstring>
// #include <time.h>
#include <string.h>
#include <ctime>
#include <cassert>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "helper_cuda.h"
#include "utils.hpp"
#include "rforest.cuh"


/* Set defaults unless user passed in arguments overriding these. */
#define NUM_POINTS 2000
#define NUM_FEATURES 5
#define NUM_TREES 1
#define SUBSAMPLING_RATIO -1
#define MAX_DEPTH 1 // NUM_POINTS

using namespace std;


class RandomForest
{
public:
    RandomForest(float *data, int num_features, int num_points, int num_trees, 
        int max_depth, float sub_sampling, bool gpu_mode);
    ~RandomForest();
    float *predict(float *x, int num_points);
    float predict_point(float *point, node *n);

    // these functions are relatively generic but are included because they contain personalized cuda versions
    float get_mse_loss(float *y, float *preds, int num_points);
    float *transpose(float *data, int num_rows, int num_cols);
    void start_time();
    float end_time();

private:
    float get_info_loss(float *y, float *col, float val, int num_points);
    node *data_split(float *data, int num_points, bool no_split, bool v);
    node *node_split(float *data, int num_points, int depth);
    void build_forest();
    float *sub_sample(float *data);

    float *data;
    int num_features;

    int num_points;
    bool gpu_mode;

    int num_trees;
    node **forest;
    int sub_sample_size;
    int max_depth;

    cublasHandle_t cublasHandle;
    float *gpu_in_x, *gpu_in_y, *gpu_out_x;
    cudaEvent_t start, stop;
};


RandomForest::RandomForest(float *data, int num_features, int num_points, 
        int num_trees, int max_depth, float sub_sampling, bool gpu_mode) {
    this->data = data;
    this->num_features = num_features; // includes y
    this->num_points = num_points;
    this->gpu_mode = gpu_mode;
    this->num_trees = num_trees; // TODO: take in from args.
    this->forest = NULL;
    this->max_depth = max_depth;
    this->sub_sample_size = (int) (sub_sampling * num_points); // TODO: fix. if -1, do not subsample. take from args

    if (gpu_mode) {
        CUBLAS_CALL(cublasCreate(&cublasHandle));

        /* Allocate arrays/matrices on gpu. Allocate with the largest size that they will be used for. */
        int size_x = (num_features - 1) * num_points;
        CUDA_CALL(cudaMalloc((void **) &this->gpu_in_x, size_x * sizeof(float)));
        CUDA_CALL(cudaMalloc((void **) &this->gpu_in_y, num_points * sizeof(float)));
        CUDA_CALL(cudaMalloc((void **) &this->gpu_out_x, size_x * sizeof(float)));
                
    }

    this->start_time();
    this->build_forest();
    printf("\nForest (%d) time: %f\n", num_trees, this->end_time());

    for (int t = 0; t < MIN(num_trees, 3); t++) {
        print_tree(this->forest[t]); cout << endl << endl;
    }
    // print_tree(this->forest[1]); cout << endl << endl;

    this->start_time();
    float *test_x = this->transpose(data + num_points, num_features - 1, num_points);
    printf("Transpose time: %f\n", this->end_time()); // (uint)(clock() - time_a)
    // print_vector(data, num_points);
    this->start_time();
    float *preds = this->predict(test_x, num_points);
    printf("Predict time: %f\n", this->end_time()); // (uint)(clock() - time_a)
    // print_vector(preds, num_points);
    this->start_time();
    printf("\tTraining loss: %f\n", this->get_mse_loss(data, preds, num_points));
    printf("Loss time: %f\n", this->end_time()); // (uint)(clock() - time_a)
}

RandomForest::~RandomForest() {
    // TODO: make sure that everything is freed
    /* Free all arrays and handles allocated and created for CUDA/CUBLAS. */
    if (this->gpu_mode) {
        CUDA_CALL(cudaFree(this->gpu_in_x));
        CUDA_CALL(cudaFree(this->gpu_in_y));
        CUDA_CALL(cudaFree(this->gpu_out_x));
        CUBLAS_CALL(cublasDestroy(this->cublasHandle));    
    }
    
    /* Free every tree of nodes in random forest. */
    if (this->forest != NULL) {
        for (int t = 0; t < this->num_trees; t++) free_tree(forest[t]);
    }
}


// with subsampling if this->sub_sample_size != -1
void RandomForest::build_forest() {
    this->forest = (node **) malloc(this->num_trees * sizeof(node *));

    float *sub;
    if (this->sub_sample_size > 0.) {
        sub = (float *) malloc(this->sub_sample_size * this->num_features * 
            sizeof(float));
        if (sub == NULL) malloc_err("build_forest");
    }

    for (int t = 0; t < this->num_trees; t++) {
        // create subsample of data
        if (this->sub_sample_size > 0.) {
            for (int s = 0; s < this->sub_sample_size; s++) {
                int p = rand() % num_points;
                for (int f = 0; f < this->num_features; f++)
                    sub[f * this->sub_sample_size + s] = this->data[f 
                        * num_points + p];
            }
            this->forest[t] = this->node_split(sub, this->sub_sample_size, 0);
        } else {
            this->forest[t] = this->node_split(this->data, this->num_points, 0);
        }
    }
    
    if (this->sub_sample_size > 0) free(sub);
}

void RandomForest::start_time() {
    CUDA_CALL(cudaEventCreate(&this->start));
    CUDA_CALL(cudaEventCreate(&this->stop));
    CUDA_CALL(cudaEventRecord(this->start));
}

float RandomForest::end_time() {
    float time = -1.;
    CUDA_CALL(cudaEventRecord(this->stop));
    CUDA_CALL(cudaEventSynchronize(this->stop));
    CUDA_CALL(cudaEventElapsedTime(&time, this->start, this->stop));
    CUDA_CALL(cudaEventDestroy(this->start));
    CUDA_CALL(cudaEventDestroy(this->stop));
    return time;
}


/* Get the information lost (impurity in y) by splitting data across col by 
val. y and col are of length num_points. This is cpu-only. */
float RandomForest::get_info_loss(float *y, float *feature, float val, int num_points) {
    float part1_y = 0.;  // sum of y where col >= val (partition 1)
    float part1_n = SMALL;  // number of rows where col >= val
    float part2_y = 0.;  //sum of y where col < val (partition 2)
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


float RandomForest::predict_point(float *point, node *n) {
    // TODO: convert this into (!n->feature)
    if (n->feature == 0)
        return n->val;
    
    if (point[n->feature - 1] >= n->val)
        return predict_point(point, n->true_branch);
    else
        return predict_point(point, n->false_branch);
}

// Takes data without y vector in point-major order (feature-major everywhere else)
// also num_features is number of columns in x.
float *RandomForest::predict(float *x, int num_points) {
    float *y = (float *) malloc(num_points * sizeof(float));
    if (y == NULL)  malloc_err("predict");

    for (int p = 0; p < num_points; p++) {
        y[p] = 0.;
        for (int t = 0; t < num_trees; t++) {
            y[p] += predict_point(x + p * (this->num_features - 1), 
                this->forest[t]);
        }
        y[p] = (y[p] * 2) > num_trees; // get mode
    }
    
    return y;
}

// calculates and returns mean-squared error loss
float RandomForest::get_mse_loss(float *y, float *preds, int num_points) {
    float loss;
    // if gpu mode and we have enough room in pre-allocated gpu_in_y and gpu_out_x, then use cublas
if (this->gpu_mode && (num_points <= this->num_points)) {
    float negone = -1.0;
    
    CUDA_CALL(cudaMemcpy(this->gpu_in_y, y, num_points * sizeof(float), 
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(this->gpu_out_x, preds, num_points * sizeof(float), 
        cudaMemcpyHostToDevice));
    
    CUBLAS_CALL(cublasSaxpy(this->cublasHandle, num_points, &negone, this->gpu_out_x, 1, 
        this->gpu_in_y, 1));
    CUBLAS_CALL(cublasSnrm2(this->cublasHandle, num_points, this->gpu_in_y, 1, &loss));
    loss *= loss; // because Snrm2 returns sqrt(sum(squares))
} 
else {
    if (this->gpu_mode) {
        printf("Warning: Could not use gpu for get_mse_loss()\n");
    }
    loss = 0.;
    for (int i = 0; i < num_points; i++) {
        loss += fabs(y[i] - preds[i]);
    }
}
    loss /= num_points;
    return loss;
}


// Returns transpose of matrix (not in-place). Uses gpu if GPU_MODE
float *RandomForest::transpose(float *data, int num_rows, int num_cols) {
    float *t = (float *) malloc(num_rows * num_cols * sizeof(float));
    if (t == NULL) malloc_err("transpose");

    int size = num_rows * num_cols;

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

/* Find optimal split in the data (by attribute and by val) that minimizes 
impurity. */
node *RandomForest::data_split(float *data, int num_points, bool no_split, bool v) {
    node *n = new node;

    /* Get unc (uncertainty/impurity) in all the data (note that data[0] is 
    in y). */
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

if(!this->gpu_mode) {
    /* Now, determine what split causes the smallest information loss. */
    n->gain = unc;
    float info_loss;

    for (int i = num_points; i < (this->num_features * num_points); i += num_points) {
        for (int p = 0; p < num_points; p++) {
            info_loss = this->get_info_loss(data, data + i, data[i + p], num_points);
            if (info_loss < n->gain) {
                n->gain = info_loss;
                n->feature = i / num_points;
                n->val = data[i + p];
            }
        }
    }

} else {
    int f = this->num_features - 1;
    int size_x = f * num_points;

    /* Copy data into inputs. */
    CUDA_CALL(cudaMemcpy(this->gpu_in_x, data + num_points, size_x * sizeof(float), 
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(this->gpu_in_y, data, num_points * sizeof(float), 
        cudaMemcpyHostToDevice));

    /* Compute information gains. */
    cuda_call_get_losses(this->gpu_in_x, this->gpu_in_y, this->gpu_out_x, f, 
        num_points);
    int result;

    CUBLAS_CALL(cublasIsamin(this->cublasHandle, size_x, this->gpu_out_x, 1, 
        &result));
    result -= 1;
    CUDA_CALL(cudaMemcpy(&n->gain, this->gpu_out_x + result, sizeof(float), 
        cudaMemcpyDeviceToHost));
    result += num_points;
    n->feature = (result / num_points);
    n->val = data[result];

}
    n->gain = unc - n->gain;

    // Set to the parition to be the mode of y, since no possible info gain
    if (n->gain <= SMALL) {
        n->feature = 0;
        n->val = (y_true * 2) > num_points;
    }

    return n;
}


node *RandomForest::node_split(float *data, int num_points, int depth) {
    /* Find the optimal split in the data. */
    // TODO: in function, use gpu based on num_points & num_features

    node *n = this->data_split(data, num_points, (depth >= this->max_depth), false);

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

    float *t_data = (float *) malloc(t_points * this->num_features * sizeof(float));
    float *f_data = (float *) malloc(f_points * this->num_features * sizeof(float));
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
    
    n->false_branch = this->node_split(f_data, f_points, depth + 1);
    free(f_data);
    n->true_branch = this->node_split(t_data, t_points, depth + 1);
    free(t_data);

    return n;
}




int main(int argc, char **argv) {
    // int num_features = NUM_FEATURES, num_points = NUM_POINTS;
    
    /* Check number of arguments and parse them. */
    if (argc < 2) {
        printf("Incorrect number of arguments passed in (%d).\n"
        "Usage: ./rforest <path of csv>\n\t[-s/--shape <# points> <# features>]"
        "\n\t[-t/--trees <# of trees>]\n\t[-d/--depth <max depth of trees>]" 
        "\n\t[-f/-frac <fraction to use for subsmapling, if <=0, no sampling>]\n", 
        argc);
        exit(0);
    }
    string path = argv[1];
    int num_points = NUM_POINTS;
    int num_features = NUM_FEATURES;
    int num_trees = NUM_TREES;
    float sub_sampling = SUBSAMPLING_RATIO;
    int max_depth = MAX_DEPTH;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--shape") == 0 || strcmp(argv[i], "-s") == 0) {
            i++;
            if (i + 1 < argc) {
                num_points = atoi(argv[i]);
                i++;
                num_features = atoi(argv[i]);
            }
        } else if (strcmp(argv[i], "--trees") == 0 || strcmp(argv[i], "-t") == 0) {
            i++;
            if (i < argc) num_trees = atoi(argv[i]);
        } else if (strcmp(argv[i], "--depth") == 0 || strcmp(argv[i], "-d") == 0) {
            i++;
            if (i < argc) max_depth = atoi(argv[i]);
        } else if (strcmp(argv[i], "--frac") == 0 || strcmp(argv[i], "-f") == 0) {
            i++;
            if (i < argc) sub_sampling = atof(argv[i]);
        }
    }


    /* Read in data from path. */
    float *data = read_csv(path, num_features, num_points, false);
    // print_matrix(data, num_features, num_points);

    /* Do benchmarking. */
    printf("\n************ CPU benchmarking: ************\n");
    RandomForest(data, num_features, num_points, num_trees, max_depth, 
        sub_sampling, false);
    
    printf("\n\n************ GPU/CUDA benchmarking: ************\n");
    RandomForest(data, num_features, num_points, num_trees, max_depth, 
      sub_sampling, true);

    free(data);

    return 0;
}


