#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Core>
#include "./config.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <assert.h>
#include <fstream>


using namespace Eigen;
using namespace std;

typedef Matrix< int, 3, Dynamic > ExampleMat;
typedef Matrix< float, N_LATENT, Dynamic > UserMat;
typedef Matrix< float, N_LATENT, Dynamic > ProductMat;
typedef Matrix< float, N_LATENT, 1 > LatentVec;
typedef Matrix< float, Dynamic, 1 > ExampleVec;
typedef Matrix< float, Dynamic, 1 > ProductVec;
typedef Matrix< float, Dynamic, 1 > UserVec;

#endif
