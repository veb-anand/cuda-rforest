/* 
 * RandomForest
 * @author Vaibhav Anand, 2018
 */

#include <fstream>
#include <iostream>
#include <string.h>

#include "utils.hpp"

using namespace std;

/****** NODE FUNCTIONS ******/
/* Print node n's members recursively. */
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

/* Print node n's members (not recursive) */
void print_node(node *n) {
    printf("Node structure: feature=%d, val=%.4f\n\n",
        n->feature, n->val);
}

/* Frees a tree by recursively freeing all of its nodes. */
void free_tree(node *n) {
    if (n->feature != 0) {
        free_tree(n->true_branch);
        free_tree(n->false_branch);
    }
    delete n;
}


/****** IO FUNCTIONS ******/
/* Prints information regarding an error in reading in csv file due to 
incorrect dimensions. */
inline void reading_err(int f, int p, int ef, int ep){
    printf("ERROR: incorrect CSV dimensions. Expected (%d, %d) " 
        "but failed at point (%d, %d)\n", ep, ef, p, f);
    exit(0);
}

/* Reads a csv file at path (extension included). It checks that the file read 
has arg:num_features and arg:num_points. It expects the file to be stored in 
comma-delimited, point-major format with y as the first column. */
float *read_csv(string path, int num_features, int num_points, bool verbose) {
    /* Allocate matrix for data. */
    float *data = (float *) malloc(num_features * num_points * sizeof(float));
    if (data == NULL) malloc_err("read_csv");

    /* Open file. */
    ifstream f(path);
    if(!f.is_open()) cout << "ERROR: File Open" << '\n';


    string line;
    char* token;
    const char delimiter[2] = ",";
    int feat, p = 0;

    /* Prase through the file, line by line, converting delimited numbers 
    into floats and storing in data. */
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

    /* Close file. */
    f.close();

    /* Print result or reading in data from file. */
    printf("\nSuccesfully read in data(points:%d, features:%d)\n", 
        num_points, num_features);

    if (verbose) {
        print_matrix(data, num_features, num_points);
        cout << endl;
    }

    return data;
}

/* Prints matrix arg:data that is in feature-major format. (Do not use on 
large datasets unless you want to print that much). */
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

/* Prints a vector of arg:data with arg:num_points. */
void print_vector(float *data, int num_points) {
    for (int i = 0; i < num_points; i++) {
        printf("%.6f ", data[i]);
    }
    cout << endl;
}

/* Prints arg:where (what function) an error caused by malloc() failing 
occurred. It causes the program to exit(1), since this is never expected. */
inline void malloc_err(string where) {
    printf("Error: malloc() in %s()\n", where.c_str());
    exit(1);
}

