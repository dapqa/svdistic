#include "../../types.h"
#include "../svd/model.h"
#include "../../utils/pipes.h"


#ifndef _SVDPP_H
#define _SVDPP_H

class TimeSVDpp : public SVDpp
{
  public:
    // Manipulate weights
    void init_weights();
    void load_weights();
    void save_weights();

  protected:

    float predict(ExampleMat& X, int ij);
    void per_corpus(ExampleMat& X);
    void per_epoch(ExampleMat& X);
    void init_user(ExampleMat& X, int ij);
    void end_user(ExampleMat& X, int ij);
    void update(ExampleMat& X, int ij);

    // Model weights
    UserVec t_mu;
    Vector t_mu;
};

#endif

