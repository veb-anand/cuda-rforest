/* 
 * Random Forest
 * Vaibhav Anand, 2018
 */

#include <cstdio>
#include <fstream>
#include <iostream>
// #include <cstdlib>
#include <cstring>
#include <string.h>

#include "utils.hpp"


/***** NODE FUNCTIONS */

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


/****** IO FUNCTIONS ******/

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

float *read_csv(string path, int num_features, int num_points, bool verbose) {
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

    if (verbose) {
        printf("\nSuccesfully read in data(features:%d, rows:%d):\n", 
            num_features, num_points);
        print_matrix(data, num_features, num_points);
        cout << endl;
    }

    return data;
}

