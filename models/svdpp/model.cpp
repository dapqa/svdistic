#include "./model.h"
#include "./helpers.cpp"


/***********************************************
 * Predict stuff
 **********************************************/
// pred = mu + b_u + b_p + W_p(W_u + |R(u)|^-1/2 sum y_j)
float SVDpp::predict(ExampleMat& X, int ij)
{
  const float i = X(0, ij);
  const float j = X(1, ij);
  return mu + b_p(j) + b_u(i) + W_p.col(j).transpose() *
      (W_u.col(i) + RuNorm(i) * Ysum.col(i));
}


/***********************************************
 * Update weights
 **********************************************/

// Update product weight: W_p = W_p + lr*(e*W_u[ui] - g * W_p)
void SVDpp::product_weight(float err, ExampleMat& X, int ij)
{
  const float i = X(0, ij);
  const float j = X(1, ij);
  W_p.col(j) += LR * (err * (W_u.col(i) + RuNorm(i) * Ysum.col(i)) -
                      REG_W * W_p.col(j));
}

// W_i = W_i + LR * (err * Runorm * W_p - REG * W_i)
void SVDpp::implicit_weight(ExampleMat &X, int i)
{
  for (int ijj = user_start_ij; ijj < ((int) user_start_ij) + ((int) Ru(i)); ++ijj)
  {
    const float jj = X(1, ijj);
    W_i.col(jj) += LR * (implicit_term - REG_W * W_i.col(jj));
  }
}

// W_i = W_i + LR * (err * Runorm * W_p - REG * W_i)
void SVDpp::accum_implicit(float err, ExampleMat& X, int ij)
{
  const float i = X(0, ij);
  const float j = X(1, ij);
  implicit_term += err * RuNorm(i) * W_p.col(j);
}

