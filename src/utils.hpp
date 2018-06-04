/* 
 * Random Forest
 * Vaibhav Anand, 2018
 */

#pragma once

#include <iostream>
#include <string.h>

#define GINI(x) (1. - x * x - (1. - x) * (1. - x))
#define MIN(a, b) (a > b ? b : a)
#define MAX(a, b) (a > b ? a : b)
#define SMALL 1e-7 // used to prevent against divide by 0 errors

using namespace std;


/***** NODE FUNCTIONS */
struct node {
    int feature; // if feature = 0, then all values are "val". no further splits.
    float val;
    float gain;
    node *false_branch;
    node *true_branch;
};

void print_node(node *n);

void print_tree(node *n);

void free_tree(node *n);

void malloc_err(string where);

/****** IO FUNCTIONS ******/
void print_matrix(float *data, int num_features, int num_points);

void print_vector(float *data, int num_points);

float *read_csv(string path, int num_features, int num_points, bool verbose);

