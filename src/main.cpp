/* 
 * Random Forest
 * Vaibhav Anand, 2018
 */

#include <cstdlib>
#include <string.h>
#include <iostream>

#include "utils.hpp"
#include "rforest.hpp"
#include "constants.hpp"

using namespace std;


// todo: move this to main.cpp
// runs through fitting, predicting, and computing loss on the same dataset
void test_random_forest(RandomForest * clf, float *data, int num_features, 
        int num_points, int num_trees, int verbosity) {

    /* Fit random forest to data. Print several of the models, depending on 
    verbosity and print the elapsed training time. */
    clf->start_time();
    clf->fit(data, num_features, num_points);
    float elapsed_time = clf->end_time();

    if (verbosity) clf->print_forest(verbosity);
    printf("\nForest (%d) time: %f\n", num_trees, elapsed_time);

    /* Take transpose of training data and save as test_x so that it is in 
    point-major format for predicting. Print elapsed time of operation. */
    clf->start_time();
    float *test_x = clf->transpose(data + num_points, num_features - 1, num_points);
    printf("Transpose time: %f\n", clf->end_time());

    /* Predict on training data and save results in preds. Print elapsed time 
    of predicting. */
    clf->start_time();
    float *preds = clf->predict(test_x, num_points);
    printf("Predict time: %f\n", clf->end_time());

    /* Compute loss from predictions on training data. Prints the training loss 
    and the elapsed time to compute it. */
    clf->start_time();
    printf("\tTraining loss: %f\n", clf->get_mse_loss(data, preds, num_points));
    printf("Loss time: %f\n", clf->end_time());
}



int main(int argc, char **argv) {
    /* Check number of arguments and parse them. */
    if (argc < 2) {
        printf("Incorrect number of arguments passed in (%d).\n"
        "Usage: ./rforest <path of csv>\n\t[-s/--shape <# points> <# features>]"
        "\n\t[-t/--trees <# of trees>]\n\t[-d/--depth <max depth of trees>]" 
        "\n\t[-f/--frac <fraction to use for subsampling, if <=0, no sampling>]"
        "\n\t[-v/--verbose <level of verbosity, default 1>]\n", argc);
        exit(0);
    }
    string path = argv[1];
    int num_points = NUM_POINTS;
    int num_features = NUM_FEATURES;
    int num_trees = NUM_TREES;
    float sub_sampling = SUBSAMPLING_RATIO;
    int max_depth = MAX_DEPTH;
    int verbosity = VERBOSITY;

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
        } else if (strcmp(argv[i], "--verbosity") == 0 || strcmp(argv[i], "-v") == 0) {
            i++;
            if (i < argc) verbosity = atoi(argv[i]);
        }
    }

    /* Read in data from path. */
    float *data = read_csv(path, num_features, num_points, false);
    // uncomment to print csv:
    // print_matrix(data, num_features, num_points);


    /* Do benchmarking on cpu and gpu versions of the model. */
    printf("\n************ CPU benchmarking: ************\n");
    RandomForest *rforest_cpu = new RandomForest(num_trees, max_depth, 
        sub_sampling, false);
    test_random_forest(rforest_cpu, data, num_features, num_points, num_trees, 
        verbosity);
    delete rforest_cpu;

    
    printf("\n\n************ GPU/CUDA benchmarking: ************\n");
    RandomForest *rforest_gpu = new RandomForest(num_trees, max_depth, 
      sub_sampling, true);
    test_random_forest(rforest_gpu, data, num_features, num_points, num_trees, 
        verbosity);
    delete rforest_gpu;

    free(data);

    return 0;
}
