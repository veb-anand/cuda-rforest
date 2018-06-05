#pragma once

/* Contains constants used as defaults for parameterizing the random forest. 
Use these defaults below unless user passes in arguments overriding these. */

/* Shape (number of points and features) in input dataset to read in. */
#define NUM_POINTS 2000
#define NUM_FEATURES 5

/* Number of decision trees to use in random forest model. */
#define NUM_TREES 1

/* Fraction of points to use to train in each tree. Points are chosen randomly. 
If SUBSAMPLING_FRAC < 0, the original dataset will not be sampled and the 
model will train on all of the training data. */
#define SUBSAMPLING_FRAC -1

/* The maximum depth that a decision tree is allowed to recurse. This can 
help avoid over-fitting and increase performance. By default, it is set to 
a BIG number such that there is effectively no depth limitation. */
#define MAX_DEPTH BIG

/* The test script in main.cpp will print VERBOSITY number of trees when the 
model is fitted. Recommended to change to 1 to inspect what the trees look 
like. The format of the trees printed are:
    (feature to split on, value "val" to split on, 
        recursive branch for points < val, 
        recursive branch for points > val)
*/
#define VERBOSITY 0

/* These barriers represent the number of points below which training no longer 
occurs on the GPU but the CPU (if gpu_mode is true). See their usage in 
rforest.cpp for context. A higher barrier means the software is less likely to 
use the GPU. */
#define GPU_BARRIER_1 750
#define GPU_BARRIER_2 50
