/* 
 * Random Forest
 * Vaibhav Anand, 20148
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <time.h>
#include <string.h>
#include <ctime>
#include <cassert>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "helper_cuda.h"
#include "rforest.cuh"

#define VERBOSE false
#define SMALL 0.00001
#define GINI(x) (1 - x * x - (1 - x) * (1 - x))

/* GPU_MODE = 
    0: use cpu for every operation
    1: use gpu for data_split (CUBLAS only), no parallelization between trees
*/
#define GPU_MODE 1

using namespace std;

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

/* Checks the passed-in arguments for validity. */
void check_args(int argc, char **argv) {
}

void print_matrix(float *data, int num_features, int num_points) {
    int i, j, index = 0;
    for (i = 0; i < num_features; i++) {
        if (i == 0) cout << "Y:\n";
        
        for (j = 0; j < num_points; j++) {
            printf("%.6f ", data[index++]);
        }

        if (i == 0) cout << "\n\nX:";
        cout << endl;
    }
}

void print_vector(float *data, int num_points) {
    for (int i = 0; i < num_points; i++) {
        printf("%.6f ", data[i]);
    }
    cout << endl;
}

float *read_csv(string path, int num_features, int num_points) {
    float *data = (float *) malloc(num_features * num_points * sizeof(float));

    ifstream f(path);

    if(!f.is_open()) std::cout << "ERROR: File Open" << '\n';

    string num_str;
    int index = 0;

    for (int i = 0; i < (num_points * num_features); i++) {
        getline(f, num_str, ',');
        data[index++] = atof(num_str.c_str());
    }

    f.close();

    if (VERBOSE) {
        printf("\nSuccesfully read in data(features:%d, rows:%d):\n", 
            num_features, num_points);
        print_matrix(data, num_features, num_points);
        cout << endl;
    }

    return data;
}


/* Get the information lost (impurity in y) by splitting data across col by 
val. y and col are of length num_points. */
float get_info_loss(float *y, float *col, float val, int num_points) {
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

// struct split{
//     float info_gain;
//     float val;
//     int feature;
// };

// void print_split(split *s) {
//     printf("Split structure:\n" 
//         "Information gain: %.4f, Split val: %.4f, Split feature: %d\n\n",
//         s->info_gain, s->val, s->feature);
// }


struct node {
    int feature; // if feature = 0, then all values are "val". no further splits.
    float val;
    float gain;
    node *false_branch;
    node *true_branch;
};

void print_node(node *n) {
    printf("Node structure: feature=%d, val=%.4f\n\n",
        n->feature, n->val);
}

void print_tree(node *n) {
    if (n->feature == 0) {
        cout << n->val;
        return;
    }
    printf("(%d, %f, ", n->feature, n->val);
    print_tree(n->true_branch);
    cout << ", ";
    print_tree(n->false_branch);
    cout << ")";
}

float predict_point(float *point, node *n) {
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
float *predict(float *x, int num_points, int num_features, node *n) {
    float *y = (float *) malloc(num_points * sizeof(float));
    for (int p = 0; p < num_points; p++) {
        y[p] = predict_point(x + p * num_features, n);
    }
    return y;
}


// Returns transpose of matrix (not in-place). Uses gpu if GPU_MODE
float *transpose(float *data, int num_rows, int num_cols) {
    float *t = (float *) malloc(num_rows * num_cols * sizeof(float));

// if (GPU_MODE) {
    // use cublasSgeam() w/alpha=1 and beta=0    
// }
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            t[j * num_rows + i] = data[i * num_cols + j];
        }
    }

    return t;
}


/* Find optimal split in the data (by attribute and by val) that minimizes 
impurity. */
node *data_split(float *data, int num_features, int num_points) {
    node *n = new node;

    /* Get unc (uncertainty/impurity) in all the data (note that data[0] is 
    in y). */
    // TODO: do this in gpu mode
    float unc = 0.;
    for (int r = 0; r < num_points; r++) {
        unc += data[r];
    }

    /* If there is no impurity, then we cannot gain info, so return. */
    if ((unc == 0.) || (unc == num_points)) { // TODO: point of potential failure
        n->feature = 0;
        n->val = data[0];
        return n;
    }
    
    unc = GINI(unc / num_points);

if(GPU_MODE == 0) {

    /* Now, determine what split causes the smallest information loss. */
    n->gain = unc;
    float info_loss;

    for (int i = num_points; i < (num_features * num_points); i += num_points) {
        for (int r = 0; r < num_points; r++) {
            info_loss = get_info_loss(data, data + i, data[i + r], num_points);
            if (info_loss < n->gain) {
                n->gain = info_loss;
                n->feature = i / num_points;
                n->val = data[i + r];
            }
        }
    }
} else {
    cublasHandle_t cublasHandle;
    CUBLAS_CALL(cublasCreate(&cublasHandle));
    int f = num_features - 1;
    int size_x = f * num_points;

    /* Allocate arrays/matrices on gpu. */
    float *gpu_in_x, *gpu_in_y, *gpu_tmp, *gpu_out_x;
    CUDA_CALL(cudaMalloc((void **) &gpu_in_x, size_x * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &gpu_in_y, num_points * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &gpu_tmp, size_x * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &gpu_out_x, size_x * sizeof(float)));

    /* Copy data into inputs. */
    CUDA_CALL(cudaMemcpy(gpu_in_x, data + num_points, size_x * sizeof(float), 
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(gpu_in_y, data, num_points * sizeof(float), 
        cudaMemcpyHostToDevice));

    /* Compute information gains. */
    cuda_call_mat_gt_y(gpu_in_x, gpu_in_y, gpu_tmp, gpu_out_x, size_x, num_points);
    int result;
    CUBLAS_CALL(cublasIsamin(cublasHandle, size_x, gpu_out_x, 1, &result));

    // float *data2 = (float *) malloc(size_x * sizeof(float));
    // CUDA_CALL(cudaMemcpy(data2, gpu_out_x, size_x * sizeof(float), 
    //     cudaMemcpyDeviceToHost));
    // print_matrix(data2, f, num_points);
    // printf("Index: %d %d %d %f\n", result, (result % f + 1), (result / f), data2[result]);
    // free(data2);


    result -= 1;
    CUDA_CALL(cudaMemcpy(&n->gain, gpu_out_x + result, sizeof(float), 
        cudaMemcpyDeviceToHost));
    result += num_points;
    n->feature = (result / num_points);
    n->val = data[result];

    CUDA_CALL(cudaFree(gpu_in_x));
    CUDA_CALL(cudaFree(gpu_in_y));
    CUDA_CALL(cudaFree(gpu_tmp));
    CUDA_CALL(cudaFree(gpu_out_x));
    CUBLAS_CALL(cublasDestroy(cublasHandle));
}
    // printf("unc, n->gain %f %f\n", unc, n->gain);
    n->gain = unc - n->gain;
    // exit(0);

    return n;
}


node *node_split(float *data, int num_features, int num_points) {
    /* Find the optimal split in the data. */
    // TODO: in function, use gpu based on num_points & num_features
    node *n = data_split(data, num_features, num_points);
    // printf("Split: %d %f %f\n", n->feature, n->val, n->gain);

    /* Split did not help increase information, so stop. */
    if (n->feature == 0) {
        return n;
    }
    if (n->gain <= 0.001) {
        // TODO: get a mean of y from inside data_split()
        n->val = 0.;
        return n;
    }
    // exit(0);


    /* Partition data according to the split. */
    int r, f, t_rows = 0, f_rows, tr = 0, fr = 0;
    float *col = data + (num_points * n->feature);

    /* Compute number of rows in each partition. */
    for (r = 0; r < num_points; r++) t_rows += (col[r] >= n->val);
    f_rows = num_points - t_rows;

    float *t_data = (float *) malloc(t_rows * num_features * sizeof(float));
    float *f_data = (float *) malloc(f_rows * num_features * sizeof(float));

    /* Poor indexing, but faster than transposing back and forth. */
    for (r = 0; r < num_points; r++) {
        if (col[r] >= n->val) {
            for (f = 0; f < num_features; f++) {
                t_data[(t_rows * f) + tr] = data[(num_points * f) + r];
            }
            tr++;
        } else {
            for (f = 0; f < num_features; f++) {
                f_data[(f_rows * f) + fr] = data[(num_points * f) + r];
            }
            fr++;
        }
    }

    // printf("Trows:%d, Frows: %d\n", t_rows, f_rows);
    // if (n->gain == .5) {
    //     printf("num: %d %d\n", num_features, num_points);
    //     print_matrix(data, num_features, num_points);
    //     print_matrix(t_data, num_features, t_rows);
    //     exit(0);
    // }

    n->false_branch = node_split(f_data, num_features, f_rows);
    free(f_data);
    n->true_branch = node_split(t_data, num_features, t_rows);
    free(t_data);

    return n;
}

int main(int argc, char **argv) {
    // vector<int> *v = new vector<int>;
    // vector<int> v = new vector<int>;

    // create_random_data(3, 5);
    int num_points = 20, num_features = 10;

    float *data = read_csv("data/data.csv", num_features, num_points);
    // print_matrix(data, num_features, num_points);
    // float *t = transpose(data, num_features, num_points);
    // print_matrix(t, num_points, num_features);
    // return 0;

    cout << "version:\n";

    clock_t time_a = clock();
    node *tree = node_split(data, num_features, num_points);
    printf("\n\tTicks: %d\n", (uint)(clock() - time_a));

    print_tree(tree); cout << endl << endl;

    printf("Predicting\n");
    print_vector(data, num_points);
    float *test_x = transpose(data + num_points, num_features - 1, num_points);
    float *pred = predict(test_x, num_points, num_features - 1, tree);
    print_vector(pred, num_points);

    // cout << "\nFinished split:\nTrue data:\n";
    // print_matrix(t_data, num_features, t_rows);
    // cout << "\nFalse data:\n";
    // print_matrix(f_data, num_features, f_rows);

    cout << "Finished!" << endl;

    return 0;
}


