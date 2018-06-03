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

inline void reading_err(int f, int p, int ef, int ep){
    printf("ERROR: incorrect CSV dimensions. Expected (%d, %d) but got point at (%d, %d)\n", ep, ef, p, f);
    exit(1);
}

/* Expects file to be stored in comma-delimitered, point-major format with y as the first column. */
float *read_csv(string path, int num_features, int num_points, bool verbose) {
    float *data = (float *) malloc(num_features * num_points * sizeof(float));

    ifstream f(path);

    if(!f.is_open()) cout << "ERROR: File Open" << '\n';

    string line;
    char* token;
    const char delimiter[2] = ",";
    int feat, p = 0;

    while(getline(f, line)) {
        token = strtok((char *) line.c_str(), delimiter);
        feat = 0;
        do {
            data[num_points * feat + p] = atof(token);
            feat++;
            token = strtok(NULL, delimiter);
        } while (token != NULL);
        if (feat != num_features) 
            reading_err(feat, p, num_features, num_points);

        p++;
    }
    if (p != num_points) 
        reading_err(feat, p, num_features, num_points);

    f.close();

    printf("\nSuccesfully read in data(features:%d, rows:%d):\n", 
        num_features, num_points);

    if (verbose) {
        print_matrix(data, num_features, num_points);
        cout << endl;
    }

    return data;
}

