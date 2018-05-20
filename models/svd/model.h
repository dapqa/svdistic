#include "../../types.h"
#include "../base/base.h"
#include "../../utils/pipes.h"


#ifndef _SVD_H
#define _SVD_H

class SVD : public Base
{
  public:
    void init_weights();
    void load_weights();
    void save_weights();

  protected:
    void user_weight(float err, ExampleMat& X, int ij);
    void product_weight(float err, ExampleMat& X, int ij);

    float predict(ExampleMat& X, int ij);
    void update(ExampleMat& X, int ij);

    UserMat W_u;
    ProductMat W_p;
};

#endif

