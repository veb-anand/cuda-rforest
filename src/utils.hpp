/* 
 * RandomForest
 * @author Vaibhav Anand, 2018
 */

#pragma once

#include <iostream>
#include <string.h>

/* Formula to calculate GINI coefficient. */
#define GINI(x) (1. - x * x - (1. - x) * (1. - x))

/* Define macros for MIN() and MAX(). */
#define MIN(a, b) (a > b ? b : a)
#define MAX(a, b) (a > b ? a : b)

/* Define small and big constants to approximate 0 and infinity while avoiding 
divide by 0 errors and hangups. */
#define SMALL 1e-7
#define BIG 1e9

using namespace std;


/****** NODE FUNCTIONS ******/

struct node {
    /* Feature to split on. If 0, then all values in this partition are "val" 
    and there are no further splits such that false_branch and true_branch are 
    meaningless.*/
    int feature;
    
    /* Value to split on. If feature=0, then it represents value for all 
    points in this partition of the data. */
    float val;

    node *false_branch;
    node *true_branch;
};

/* Print node n's members recursively. */
void print_tree(node *n);

/* Print node n's members (not recursive) */
void print_node(node *n);

/* Frees a tree by recursively freeing all of its nodes. */
void free_tree(node *n);


/****** I/O FUNCTIONS ******/

/* Reads a csv file at path (extension included). It checks that the file read 
has arg:num_features and arg:num_points. It expects the file to be stored in 
comma-delimited, point-major format with y as the first column. */
float *read_csv(string path, int num_features, int num_points, bool verbose);

/* Prints matrix arg:data that is in feature-major format. (Do not use on 
large datasets unless you want to print that much). */
void print_matrix(float *data, int num_features, int num_points);

/* Prints a vector of arg:data with arg:num_points. */
void print_vector(float *data, int num_points);

/* Prints arg:where (what function) an error caused by malloc() failing 
occurred. It causes the program to exit(1), since this is never expected. */
void malloc_err(string where);

