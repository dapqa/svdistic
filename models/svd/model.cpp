#include "model.h"


// Calculate Pred_terms: 
// p = u + b_p + b_u + W_p^T * Q_p 
float SVD::predict(ExampleMat& X, int ij)
{
  const int i = X(0, ij);
  const int j = X(1, ij);
  return mu + b_p(j) + b_u(i) + W_p.col(j).transpose() * W_u.col(i);
}

// Update for a single element.
void SVD::update(ExampleMat& X, int ij)
{
  float err = pred_diff(X, ij);
  user_bias(err, X, ij);
  product_bias(err, X, ij);
  user_weight(err, X, ij);
  product_weight(err, X, ij);
}

// Update user weight: W_u = W_u + lr*(e * W_p[pi] - g * W_u)
// Calculate for a specific example.
void SVD::user_weight(float err, ExampleMat& X, int ij)
{
  const int i = X(0, ij);
  const int j = X(1, ij);
  W_u.col(i) += LR * (err * W_p.col(j) - REG_W * W_u.col(i));
}


// Update product weight: W_p = W_p + lr*(e*W_u[ui] - g * W_p)
// Calculate for a specific example.
void SVD::product_weight(float err, ExampleMat& X, int ij)
{
  const int i = X(0, ij);
  const int j = X(1, ij);
  W_p.col(j) += LR * (err * W_u.col(i) - REG_W * W_p.col(j));
}

// Initialize all weights.
void SVD::init_weights()
{
  b_u.setZero(N_USER);
  b_p.setZero(N_PRODUCT);
  W_u.setRandom(N_LATENT, N_USER);
  W_p.setRandom(N_LATENT, N_PRODUCT);
}

// Save weights into file.
void SVD::save_weights()
{
  save_matrix<float, N_LATENT, -1>(W_u, "data/saves/" + model_id + "-W_u");
  save_matrix<float, N_LATENT, -1>(W_p, "data/saves/" + model_id + "-W_p");
  save_float(mu, "data/saves/" + model_id + "-mu");
  save_matrix<float, -1, 1>(b_u, "data/saves/" + model_id + "-b_u");
  save_matrix<float, -1, 1>(b_p, "data/saves/" + model_id + "-b_p");
}
// Load weights from file.
void SVD::load_weights()
{
  load_matrix<float, N_LATENT, -1>(W_u, "data/saves/" + model_id + "-W_u", N_LATENT, N_USER);
  load_matrix<float, N_LATENT, -1>(W_p, "data/saves/" + model_id + "-W_p", N_LATENT, N_PRODUCT);
  mu = load_float("data/saves/" + model_id + "-mu");
  load_matrix<float, -1, 1>(b_u, "data/saves/" + model_id + "-b_u", N_USER, 1);
  load_matrix<float, -1, 1>(b_p, "data/saves/" + model_id + "-b_p", N_PRODUCT, 1);
}

