#include "./model.h"
#include "./helpers.cpp"


/***********************************************
 * Predict
 **********************************************/

// p = u + b_p + b_u + W_p^T * Q_p 
float SVD::predict(ExampleMat& X, int ij)
{
  const float i = X(0, ij);
  const float j = X(1, ij);
  return mu + b_p(j) + b_u(i) + W_p.col(j).transpose() * W_u.col(i);
}


/***********************************************
 * Update weights
 **********************************************/

// Update user weight: W_u = W_u + lr*(e * W_p[pi] - g * W_u)
void SVD::user_weight(float err, ExampleMat& X, int ij)
{
  const float i = X(0, ij);
  const float j = X(1, ij);
  W_u.col(i) += LR * (err * W_p.col(j) - REG_W * W_u.col(i));
}


// Update product weight: W_p = W_p + lr*(e*W_u[ui] - g * W_p)
// Calculate for a specific example.
void SVD::product_weight(float err, ExampleMat& X, int ij)
{
  const float i = X(0, ij);
  const float j = X(1, ij);
  W_p.col(j) += LR * (err * W_u.col(i) - REG_W * W_p.col(j));
}

