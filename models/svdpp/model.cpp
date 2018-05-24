#include "./model.h"
#include "./helpers.cpp"


/***********************************************
 * Predict stuff
 **********************************************/
// pred = mu + b_u + b_p + W_p(W_u + |R(u)|^-1/2 sum y_j)
float SVDpp::predict(ExampleMat& X, int ij)
{
  const int i = X(0, ij);
  const int j = X(1, ij);
  return mu + b_p(j) + b_u(i) + W_p.col(j).transpose() *
      (W_u.col(i) + Ru(i) * Ysum.col(i));
}


/***********************************************
 * Update weights
 **********************************************/

// Update product weight: W_p = W_p + lr*(e*W_u[ui] - g * W_p)
void SVDpp::product_weight(float err, ExampleMat& X, int ij)
{
  const int i = X(0, ij);
  const int j = X(1, ij);
  W_p.col(j) += LR * (err * (W_u.col(i) + Ru(i) * Ysum.col(i)) - REG_W * W_p.col(j));
}


// Accumulate implicit terms: += err * |R(u)|^(-1/2) * W_p
void SVDpp::accum_implicit(float err, ExampleMat& X, int ij)
{
  const int i = X(0, ij);
  const int j = X(1, ij);
  implicit_terms += err * Ru(i) * W_p.col(j);
}


// Update implicit weight for all users:
// W_i = W_i + LR * (implicit_terms - REG * W_i)
void SVDpp::implicit_weight(int j)
{
  W_i.col(j) += LR * (implicit_terms - REG_W * W_i.col(j));
}



