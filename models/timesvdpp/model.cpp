#include "./model.h"
#include "./helpers.cpp"


/***********************************************
 * Predict stuff
 **********************************************/
// pred = mu + b_u + b_p + W_p(W_u + |R(u)|^-1/2 sum y_j)
float TimeSVDpp::predict(ExampleMat& X, int ij)
{
  const float i = X(0, ij);
  const float j = X(1, ij);
  const float t = X(3, ij);
  return mu + b_p(j) + b_u(i) + W_p.col(j).transpose() *
      (W_u.col(i) + RuNorm(i) * Ysum.col(i)) + alpha(i) *
      dev_u(t) + b_u_t(t, i) + b_p_bin(t, i)
}


/***********************************************
 * Update weights
 **********************************************/

UserVec t_mu;
UserVec alpha;

ExampleVec dev_u;

MatrixXd b_p_bin(30, N_PRODUCT);



void calc_t_mu(ExampleMat& X)
void alpha(ExampleMat& X, int ij)

void calc_dev(ExampleMat& X)
void b_p_bin(ExampleMat& X, int ij)

