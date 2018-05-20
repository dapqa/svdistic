#include "../../types.h"
#include "../svd/model.h"
#include "../../utils/pipes.h"


#ifndef _SVDPP_H
#define _SVDPP_H

class SVDpp : public SVD
{
  public:
    void init_weights();
    void load_weights();
    void save_weights();
    void train(ExampleMat& X_tr);
    void infer(ExampleVec& preds, ExampleMat& X_test);
    float score(ExampleMat& X_val);

  protected:
    void product_weight(float err, ExampleMat& X, int ij);
    void accum_implicit(float err, ExampleMat& X, int ij);
    void implicit_weight(int j);

    float predict(ExampleMat& X, int ij);

    ProductMat W_i;
    LatentVec implicit_terms;

    int user_start_ij;
    UserMat Ysum;
    UserVec Ru;

    void per_corpus(ExampleMat& X);
    void per_epoch(ExampleMat& X);
    void per_user(ExampleMat& X, int ij);
    void update(ExampleMat& X, int ij);
    
};

#endif

