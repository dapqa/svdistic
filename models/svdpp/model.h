#include "../../types.h"
#include "../svd/model.h"
#include "../../utils/pipes.h"


#ifndef _SVDPP_H
#define _SVDPP_H

class SVDpp : public SVD
{
  public:
    // Manipulate weights
    void init_weights();
    void load_weights();
    void save_weights();

  protected:
    // Update weights
    void product_weight(float err, ExampleMat& X, int ij);
    void implicit_weight(ExampleMat& X, int i);
    void accum_implicit(float err, ExampleMat&, int ij);

    float predict(ExampleMat& X, int ij);
    void per_corpus(ExampleMat& X);
    void per_epoch(ExampleMat& X);
    void init_user(ExampleMat& X, int ij);
    void end_user(ExampleMat& X, int ij);
    void update(ExampleMat& X, int ij);

    // Model weights
    ProductMat W_i;
    int user_start_ij;
    int scale;
    UserMat Ysum;
    UserVec Ru;
    UserVec RuNorm;
    LatentVec implicit_term;
};

#endif

