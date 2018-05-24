#include "../../types.h"
#include "../base/base.h"
#include "../../utils/pipes.h"


#ifndef _SVD_H
#define _SVD_H

class SVD : public Base
{
  public:
    // Manipulate weights
    void init_weights();
    void load_weights();
    void save_weights();

  protected:
    // Update weights
    void user_weight(float err, ExampleMat& X, int ij);
    void product_weight(float err, ExampleMat& X, int ij);

    // Predict and update
    float predict(ExampleMat& X, int ij);
    void per_corpus(ExampleMat& X);
    void per_epoch(ExampleMat& X);
    void per_user(ExampleMat& X, int ij);
    void update(ExampleMat& X, int ij);

    // Model weights
    UserMat W_u;
    ProductMat W_p;
};

#endif

