#include "../types.h"
#include "../svdpp/svdpp.h"
#include "../utils/pipes.h"


#ifndef _TIMESVDPP_H
#define _TIMESVDPP_H

class TimeSVDpp : public SVDpp
{
  public:
    void init_weights();
    void load_weights();
    void save_weights();

  protected:
    float predict(ExampleMat& X, int ij);

  private:
    UserVec a_u;
    UserVec b_u_t;
    ProductVec b_p_t;

    void corpus_init(ExampleMat& X);
    void epoch_init(ExampleMat& X);
    void user_init();
    void update(ExampleMat& X, int ij);
    
};

#endif

