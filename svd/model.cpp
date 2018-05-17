#include "./model.h"


// Train function for SVD++
void train(ExampleMat& X_tr, UserMat& W_u, ProductMat& W_p)
{
  // Initialize counters.
  // i: user id
  // j: product id
  int i, j;

  // Calculate Err_terms.
  ExampleVec Err_terms = ExampleVec::Zero();
  calc_Err_terms(Err_terms, X_tr, W_u, W_p);
  cout << "RMSE: " << Err_terms.squaredNorm() << endl;

  // Iterate through all examples.
  for (int ij = 0; ij < N_EXAMPLE; ++ij)
  {
    // Grab user id and product id.
    i = X_tr(ij, 0);
    j = X_tr(ij, 1);

    // Update W_u, W_p for a user
    update_user(W_u, W_p, Err_terms(ij), i, j);
    update_product(W_u, W_p, Err_terms(ij), i, j);
  }
}

////////////////////////////////////////////////////////////
// Initialization
////////////////////////////////////////////////////////////

// Initialize all weights between -1 and 1
void init_weights(UserMat& W_u, ProductMat& W_p)
{
  W_u = UserMat::Random(N_LATENT, N_USER);
  W_p = ProductMat::Random(N_LATENT, N_PRODUCT);
}

////////////////////////////////////////////////////////////
// Calc for all users.
////////////////////////////////////////////////////////////

// Calculate Pred_terms: W_p.col(j).transpose() *
//   (W_u.col(i) + Ru_terms(i) * Ysum_terms.col(i)); 
// Calculate for all examples.
void calc_Pred_terms(ExampleVec& Pred_terms, ExampleMat& X_tr, UserMat& W_u,
                     ProductMat& W_p)
{
  int i, j;
  for (int ij = 0; ij < N_EXAMPLE; ++ij)
  {
    i = X_tr(ij, 0);
    j = X_tr(ij, 1);
    // Pred(i, j) = p_j * (u_i)
    Pred_terms(ij) = W_p.col(j).transpose() * W_u.col(i);
  }
}

// Calculate Err_terms: err_ij = (y - pred)^2
// Calculate for all examples.
void calc_Err_terms(ExampleVec& Err_terms, ExampleMat& X_tr, UserMat& W_u,
                    ProductMat& W_p)
{
  ExampleVec Pred_terms;
  calc_Pred_terms(Pred_terms, X_tr, W_u, W_p);
  Err_terms = (X_tr.col(2).cast <float> ()) - Pred_terms;
}


////////////////////////////////////////////////////////////
// Update rules
////////////////////////////////////////////////////////////

// Update user weight: W_u = W_u + lr*(e * W_p[pi] - g * W_u)
// Calculate for a specific example.
void update_user(UserMat& W_u, ProductMat& W_p, float err, int i, int j)
{
  W_u.col(i) += LR * ((err * W_p.col(j)) - (GAMMA * W_u.col(i)));
}

// Update product weight: W_p = W_p + lr*(e*W_u[ui] - g * W_p)
// Calculate for a specific example.
void update_product(UserMat& W_u, ProductMat& W_p, float err, int i, int j)
{
  W_p.col(j) += LR * ((err * W_u.col(i)) - (GAMMA * W_p.col(j)));
}

