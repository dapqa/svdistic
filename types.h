#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Core>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <assert.h>
#include <fstream>


static const int N_EPOCHS = 500;
static const int N_PRODUCT = 3;
static const int N_USER = 4;
static const int N_LATENT = 2;
static const int N_EXAMPLE = 11;

static const int EMPTY_VAL = 0;

static const float GAMMA = 0.05;
static const float LR = 0.005;


using namespace Eigen;
using namespace std;


// For storing training data.
typedef Matrix< int, Dynamic, 4 > ExampleMat;

// Store W_u
typedef Matrix< float, Dynamic, Dynamic > UserMat;
// Store W_p and W_i
typedef Matrix< float, Dynamic, Dynamic > ProductMat;

// Vectors
typedef Matrix< float, N_LATENT, 1 > LatentVec;
typedef Matrix< float, N_EXAMPLE, 1 > ExampleVec;
typedef Matrix< float, N_PRODUCT, 1 > ProductVec;
typedef Matrix< float, N_USER, 1 > UserVec;

#endif
