#pragma once

/* Set defaults unless user passed in arguments overriding these. */
#define NUM_POINTS 1000
#define NUM_FEATURES 5
#define NUM_TREES 1
#define SUBSAMPLING_RATIO -1
#define MAX_DEPTH NUM_POINTS // effectively no depth limitation when MAX_DEPTH = NUM_POINTS
#define VERBOSITY 0 // TODO: change to 1

// gpu point barriers: number of points needed at different stages to switch from using cpu to gpu (note: based on benchmarking on a single PC). see their usage for context. higher barrier means less likely to use the gpu
#define GPU_BARRIER_1 750
#define GPU_BARRIER_2 50
