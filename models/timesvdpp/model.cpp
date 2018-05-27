#include "./model.h"
#include "./helpers.cpp"


/***********************************************
 * Predict stuff
 **********************************************/
// pred = mu + b_u + b_p + W_p(W_u + |R(u)|^-1/2 sum y_j)
float TimeSVDpp::predict(ExampleMat& X, int ij)
{
  const int i = X(0, ij);
  const int j = X(1, ij);
  return mu + b_p(j) + b_u(i) + W_p.col(j).transpose() *
      (W_u.col(i) + RuNorm(i) * Ysum.col(i));
}


/***********************************************
 * Update weights
 **********************************************/


