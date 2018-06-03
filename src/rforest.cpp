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


/* GPU_MODE = 
    0: use cpu for every operation
    1: use gpu for data_split (CUBLAS only), no parallelization between trees
*/
#define GPU_MODE true
#define NUM_POINTS 1000
#define NUM_FEATURES 50

using namespace std;



class RandomForest
{
public:
    RandomForest(float *data, int num_features, int num_points, bool gpu_mode);
    ~RandomForest();
    float *predict(float *x, int num_points, node *n);
    float predict_point(float *point, node *n);

    // these functions are relatively generic but are included because they contain personalized cuda versions
    float get_mse_loss(float *y, float *preds, int num_points);
    float *transpose(float *data, int num_rows, int num_cols);
    void start_time();
    float end_time();

protected:
    float get_info_loss(float *y, float *col, float val, int num_points);
    node *data_split(float *data, int num_points, bool v);
    node *node_split(float *data, int num_points);

    float *data;
    int num_features;
    int num_points;
    bool gpu_mode;

    cublasHandle_t cublasHandle;
    float *gpu_in_x, *gpu_in_y, *gpu_tmp, *gpu_out_x;
    cudaEvent_t start, stop;
};


RandomForest::RandomForest(float *data, int num_features, int num_points, 
        bool gpu_mode) {
    this->data = data;
    this->num_features = num_features; // includes y
    this->num_points = num_points;
    this->gpu_mode = gpu_mode;

    CUBLAS_CALL(cublasCreate(&cublasHandle));

    /* Allocate arrays/matrices on gpu. Allocate with the largest size that they will be used for. */
    int size_x = (num_features - 1) * num_points;
    CUDA_CALL(cudaMalloc((void **) &this->gpu_in_x, size_x * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &this->gpu_in_y, num_points * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &this->gpu_tmp, size_x * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &this->gpu_out_x, size_x * sizeof(float)));

    // print_matrix(data + num_points, num_features - 1, num_points);
    // float *t = transpose(data + num_points, num_features - 1, num_points);
    // print_matrix(t, num_points, num_features - 1);
    // return;

    if (gpu_mode) printf("GPU/CUDA benchmarking:\n");
    else          printf("CPU benchmarking:\n");

    // clock_t time_a = clock();
    this->start_time();
    node *tree = this->node_split(data, num_points);

    print_tree(tree); cout << endl << endl;
    printf("\nTree time: %f\n", this->end_time()); // (uint)(clock() - time_a)

    this->start_time();
    float *test_x = this->transpose(data + num_points, num_features - 1, num_points);
    printf("\nTranspose time: %f\n", this->end_time()); // (uint)(clock() - time_a)
    // print_vector(data, num_points);
    float *preds = this->predict(test_x, num_points, tree);
    // print_vector(preds, num_points);
    this->start_time();
    printf("\n\tTraining loss: %f\n", this->get_mse_loss(data, preds, num_points));
    printf("\nLoss time: %f\n", this->end_time()); // (uint)(clock() - time_a)

    cout << "Finished!" << endl;

}

RandomForest::~RandomForest() {
    // TODO: make sure that everything is freed
    free(this->data);

    CUDA_CALL(cudaFree(this->gpu_in_x));
    CUDA_CALL(cudaFree(this->gpu_in_y));
    CUDA_CALL(cudaFree(this->gpu_tmp));
    CUDA_CALL(cudaFree(this->gpu_out_x));
    CUBLAS_CALL(cublasDestroy(this->cublasHandle));
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
float RandomForest::get_info_loss(float *y, float *col, float val, int num_points) {
    float part1_y = 0.;  // sum of y where col >= val (partition 1)
    float part1_n = SMALL;  // number of rows where col >= val
    float part2_y = 0.;  //sum of y where col < val (partition 2)
    float part2_n = SMALL;  // number of rows where col < val

    for (int r = 0; r < num_points; r++) {
        if (col[r] >= val) {
            part1_y += y[r];
            part1_n += 1.;
        } else {
            part2_y += y[r];
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
    if (n->feature == 0) {
        return n->val;
    }
    if (point[n->feature - 1] >= n->val) {
        return predict_point(point, n->true_branch);
    } else {
        return predict_point(point, n->false_branch);
    }
}

// Takes data without y vector in point-major order (feature-major everywhere else)
// also num_features is number of columns in x.
float *RandomForest::predict(float *x, int num_points, node *n) {
    float *y = (float *) malloc(num_points * sizeof(float));
    for (int p = 0; p < num_points; p++) {
        y[p] = predict_point(x + p * (this->num_features - 1), n);
    }
    return y;
}

// calculates and returns mean-squared error loss
float RandomForest::get_mse_loss(float *y, float *preds, int num_points) {
    float loss;
    // if gpu mode and we have enough room in pre-allocated gpu_in_y and gpu_tmp, then use cublas
if ((this->gpu_mode) && (num_points <= this->num_points)) {
    float negone = -1.0;
    CUDA_CALL(cudaMemcpy(this->gpu_in_y, y, num_points * sizeof(float), 
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(this->gpu_tmp, preds, num_points * sizeof(float), 
        cudaMemcpyHostToDevice));
    CUBLAS_CALL(cublasSaxpy(this->cublasHandle, num_points, &negone, this->gpu_tmp, 1, 
        this->gpu_in_y, 1));
    CUBLAS_CALL(cublasSnrm2(this->cublasHandle, num_points, this->gpu_in_y, 1, &loss));
    loss *= loss;
} 
else {
    if (this->gpu_mode) {
        printf("Warning: Could not use gpu for get_mse_loss()\n");
    }
    loss = 0.;
    for (int i = 0; i < num_points; i++) {
        loss += (y[i] - preds[i]) * (y[i] - preds[i]);
    }
}
    loss /= num_points;
    return loss;
}


// Returns transpose of matrix (not in-place). Uses gpu if GPU_MODE
float *RandomForest::transpose(float *data, int num_rows, int num_cols) {
    float *t = (float *) malloc(num_rows * num_cols * sizeof(float));
    int size = num_rows * num_cols;

if ((this->gpu_mode) && ((this->num_features - 1) * this->num_points >= size)) {
    float one = 1., zero = 0.;
    CUDA_CALL(cudaMemcpy(this->gpu_tmp, data, size * sizeof(float), 
        cudaMemcpyHostToDevice));
    // use cublasSgeam() w/alpha=1 and beta=0
    CUBLAS_CALL(cublasSgeam(this->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_rows, num_cols, &one, this->gpu_tmp, num_cols, &zero, this->gpu_tmp, num_rows, this->gpu_out_x, num_rows));
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
node *RandomForest::data_split(float *data, int num_points, bool v) {
    node *n = new node;

    /* Get unc (uncertainty/impurity) in all the data (note that data[0] is 
    in y). */
    // TODO: do this in gpu mode
    float unc = 0.;
    for (int r = 0; r < num_points; r++) {
        unc += data[r];
    }

    /* If there is no impurity, then we cannot gain info, so return. */
    if ((unc < 1.) || (unc > num_points - 1)) { // TODO: point of potential failure
        n->feature = 0;
        n->val = (unc > num_points - 1);
        return n;
    }
    
    unc = GINI(unc / num_points);

if(!this->gpu_mode) {
    /* Now, determine what split causes the smallest information loss. */
    n->gain = unc;
    float info_loss;

    for (int i = num_points; i < (this->num_features * num_points); i += num_points) {
        for (int r = 0; r < num_points; r++) {
            info_loss = this->get_info_loss(data, data + i, data[i + r], num_points);
            if (info_loss < n->gain) {
                // printf("i%d, loss%f val%f\n", i, info_loss, data[i+r]);
                n->gain = info_loss;
                n->feature = i / num_points;
                n->val = data[i + r];
            }
        }
    }
    // if (n->gain < 0.) {
    //     printf("unc %f, n->gain %f\n", unc, n->gain);
    //     this->get_info_loss(data, data + (n->feature * num_points), n->val, 1);
    //     print_vector(data, num_points);
    //     exit(0);
    // }

} else {
    int f = this->num_features - 1;
    int size_x = f * num_points;

    /* Copy data into inputs. */
    CUDA_CALL(cudaMemcpy(this->gpu_in_x, data + num_points, size_x * sizeof(float), 
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(this->gpu_in_y, data, num_points * sizeof(float), 
        cudaMemcpyHostToDevice));

    /* Compute information gains. */
    cuda_call_get_losses(this->gpu_in_x, this->gpu_in_y, this->gpu_tmp, 
        this->gpu_out_x, f, num_points);
    int result;

    if (v) {
        float *data2 = (float *) malloc(size_x * sizeof(float));
        CUDA_CALL(cudaMemcpy(data2, this->gpu_out_x, size_x * sizeof(float), 
            cudaMemcpyDeviceToHost));
        // print_matrix(data, f, 5);
        // print_matrix(data2, f, 5);
        // printf("Index: %d %d %d %f %d\n", result, (result % f + 1), (result / f), data2[result], num_points);
        for (int i = num_points; i < (this->num_features * num_points); i += num_points) {
            for (int r = 0; r < 5; r++) {
                int info_loss = this->get_info_loss(data, data + i, data[i + r], num_points);
                printf("cpu: %f, gpu:%f\n", info_loss, data2[i + r - num_points]);
            }
        }
        free(data2);
    }

    CUBLAS_CALL(cublasIsamin(this->cublasHandle, size_x, this->gpu_out_x, 1, 
        &result));
    result -= 1;
    CUDA_CALL(cudaMemcpy(&n->gain, this->gpu_out_x + result, sizeof(float), 
        cudaMemcpyDeviceToHost));
    result += num_points;
    n->feature = (result / num_points);
    n->val = data[result];

}
    if (v) printf("unc, n->gain %f %f\n", unc, n->gain);
    n->gain = unc - n->gain;
    if (v) printf("unc, n->gain %f %f\n", unc, n->gain);
    // exit(0);

    return n;
}


node *RandomForest::node_split(float *data, int num_points) {
    /* Find the optimal split in the data. */
    // TODO: in function, use gpu based on num_points & num_features
    // node *n = this->data_split(data, num_points, false);
    // printf("Split: %d %f %f, %d\n", n->feature, n->val, n->gain, num_points);

    // this->gpu_mode = false;
    // node *n = this->data_split(data, num_points, false);
    // printf("C-Split: %d %f %f, %d\n", n->feature, n->val, n->gain, num_points);
    // this->gpu_mode = true;
    n = this->data_split(data, num_points, false);
    // printf("G-Split: %d %f %f, %d\n", n->feature, n->val, n->gain, num_points);

    /* Split did not help increase information, so stop. */
    if (n->feature == 0) {
        return n;
    }
    if (n->gain <= 0.000) {
        // TODO: get a mean of y from inside data_split()
        n->val = 0.;
        n->feature = 0;
        return n;
    }

    /* Partition data according to the split. */
    int t_points = 0, f_points, tr = 0, fr = 0;
    float *col = data + (num_points * n->feature);

    /* Compute number of rows in each partition. */
    for (int r = 0; r < num_points; r++) t_points += (col[r] >= n->val);
    f_points = num_points - t_points;

    float *t_data = (float *) malloc(t_points * this->num_features * sizeof(float));
    float *f_data = (float *) malloc(f_points * this->num_features * sizeof(float));

    /* Poor indexing, but faster than transposing back and forth. */
    for (int r = 0; r < num_points; r++) {
        if (col[r] >= n->val) {
            for (int f = 0; f < this->num_features; f++) {
                t_data[(t_points * f) + tr] = data[(num_points * f) + r];
            }
            tr++;
        } else {
            for (int f = 0; f < this->num_features; f++) {
                f_data[(f_points * f) + fr] = data[(num_points * f) + r];
            }
            fr++;
        }
    }
    // if ((f_points == 0) || (t_points == 0)) {
    //     printf("Num rows: %d, %d, %d, %d\n", num_points, t_points, f_points, n->feature);
    //     n = this->data_split(data, num_points, true);
    //     printf("Split: %d %f %f, %d\n", n->feature, n->val, n->gain, num_points);
    //     exit(0);
    //     // print_matrix(f_data, this->num_features, f_points);
    //     n->feature = 0;
    //     // return n;
    // }

    n->false_branch = this->node_split(f_data, f_points);
    free(f_data);
    n->true_branch = this->node_split(t_data, t_points);
    free(t_data);

    return n;
}




/* Checks the passed-in arguments for validity. */
void check_args(int argc, char **argv) {
}


int main(int argc, char **argv) {
    // create_random_data(3, 5);
    int num_features = NUM_FEATURES, num_points = NUM_POINTS;
    float *data = read_csv("data/data.csv", num_features, num_points, false);
    RandomForest(data, num_features, num_points, GPU_MODE);

    return 0;
}


