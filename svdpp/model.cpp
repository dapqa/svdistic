#include "./model.h"


// Train function for SVD++
void train(ExampleMat& X_tr, UserMat& W_u, ProductMat& W_i, ProductMat& W_p,
           UserVec& Ru_terms, UserMat& Ysum_terms)
{
  // Initialize counters.
  // i: user id
  // j: product id
  // k: user index (index for product_seen)
  int i, j, k;

  // Calculate Ysum_terms.
  calc_Ysum_terms(Ysum_terms, W_i, X_tr);

  // Calculate Err_terms.
  ExampleVec Err_terms = ExampleVec::Zero();
  calc_Err_terms(Err_terms, X_tr, W_u, W_p, Ysum_terms, Ru_terms);
  cout << "RMSE: " << Err_terms.squaredNorm() << endl;

  // Initialize per-user vectors.
  LatentVec implicit_temp = LatentVec::Zero();
  ProductVec product_seen = ProductVec::Constant(-1);
  k = 0;

  // Iterate through all examples.
  for (int ij = 0; ij < N_EXAMPLE; ++ij)
  {
    // Grab user id and product id.
    i = X_tr(ij, 0);
    j = X_tr(ij, 1);

    // Update W_u, W_p for a user
    update_user(W_u, W_p, Err_terms(ij), i, j);
    update_product(W_u, W_p, Ysum_terms, Ru_terms(i),
                   Err_terms(ij), i, j);

    // Calculate implicit term for this product of the current user.
    calc_implicit_temp(implicit_temp, Err_terms(ij),
                       Ru_terms(i), W_p, j);
    product_seen(k) = j;
    ++k;

    // If this is the last example for the user
    if (X_tr(ij, 3) == 1)
    {
      // For all products of this user
      for (int kk = 0; kk < k; ++kk)
      {
        // Recall that product_seen(kk) = product_id.
        assert(product_seen(kk) != -1);
        // We also pass k as a scaling factor to update_implicit.
        update_implicit(W_i, implicit_temp, product_seen(kk), k);
      }

      // Reset per-user variables.
      implicit_temp = LatentVec::Zero();
      product_seen = ProductVec::Constant(-1);
      k = 0;
    }
  }
}

////////////////////////////////////////////////////////////
// Initialization
////////////////////////////////////////////////////////////

// Initialize all weights between -1 and 1
void init_weights(UserMat& W_u, ProductMat& W_i, ProductMat& W_p)
{
  W_u = UserMat::Random(N_LATENT, N_USER);
  W_i = ProductMat::Random(N_LATENT, N_PRODUCT);
  W_p = ProductMat::Random(N_LATENT, N_PRODUCT);
}

////////////////////////////////////////////////////////////
// Calc for all users.
////////////////////////////////////////////////////////////

// Calculate Ru_term: |R(u)|^(-1/2) for all users.
// Calculate for all users.
void calc_Ru_terms(UserVec& Ru_terms, ExampleMat& X_tr)
{
  int i;
  for (int ij = 0; ij < N_EXAMPLE; ++ij)
  {
    // We count the number of reviews for each user
    // by incrementing the user for each example.
    i = X_tr(ij, 0);
    ++Ru_terms(i);
  }
  // We scale Ru: Ru^(-1/2) = 1/Ru^(1/2)
  Ru_terms = Ru_terms.cwiseSqrt().cwiseInverse();
}

// Calculate Ysum_terms: Sum_{j \in R(u)} y_j.
// Calculate for all users.
void calc_Ysum_terms(UserMat& Ysum_terms, ProductMat& W_i, ExampleMat& X_tr)
{
  // Initialize sum to 0.
  int i, j;

  // Cycle through all examples
  for (int ij = 0; ij < N_EXAMPLE; ++ij)
  {
    // Get user, product pair.
    i = X_tr(ij, 0);
    j = X_tr(ij, 1);
    // Update user y_sum with a product's implicit term.
    Ysum_terms.col(i) += W_i.col(j);
  }
}

// Calculate Pred_terms: W_p.col(j).transpose() *
//   (W_u.col(i) + Ru_terms(i) * Ysum_terms.col(i)); 
// Calculate for all examples.
void calc_Pred_terms(ExampleVec& Pred_terms, ExampleMat& X_tr, UserMat& W_u,
                    ProductMat& W_p, UserMat& Ysum_terms, UserVec& Ru_terms)
{
  int i, j;
  for (int ij = 0; ij < N_EXAMPLE; ++ij)
  {
    i = X_tr(ij, 0);
    j = X_tr(ij, 1);
    // Pred(i, j) = p_j * (u_i + Ru_i * Ysum_i)
    Pred_terms(ij) = W_p.col(j).transpose() *
                     (W_u.col(i) + Ru_terms(i) * Ysum_terms.col(i));
  }
}

// Calculate Err_terms: err_ij = (y - pred)^2
// Calculate for all examples.
void calc_Err_terms(ExampleVec& Err_terms, ExampleMat& X_tr, UserMat& W_u,
                    ProductMat& W_p, UserMat& Ysum_terms, UserVec& Ru_terms)
{
  ExampleVec Pred_terms;
  calc_Pred_terms(Pred_terms, X_tr, W_u, W_p, Ysum_terms, Ru_terms);
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

// Update product weight: W_p = W_p + lr*(e*(W_u[ui]+RU_norm*y_sum) - g*W_p)
// Calculate for a specific example.
void update_product(UserMat& W_u, ProductMat& W_p, UserMat& Ysum,
                    float Ru, float err, int i, int j)
{
  W_p.col(j) += LR * (
      (err * (W_u.col(i) + Ru * Ysum.col(i)))
      - (GAMMA * W_p.col(j))
    );
}

////////////////////////////////////////////////////////////
// Implicit
////////////////////////////////////////////////////////////

// Implicit_temp: e_ui * |R(u)|^(-1/2) * W_p_j
// Calculate for a specific example.
void calc_implicit_temp(LatentVec& implicit_temp, float err, float Ru,
                        ProductMat& W_p, int j)
{
  implicit_temp += (err * Ru * W_p.col(j));
}

// Update implicit weight: W_i = W_i + lr*(e*RU_norm*W_p[product_id]
//                                         - GAMMA * W_i)
// Calculate for a user.
void update_implicit(ProductMat& W_i, LatentVec& implicit_temp,
                     int j, float scale)
{
  // Note we scale up lr
  W_i.col(j) += LR * (implicit_temp - scale * GAMMA * W_i.col(j));
}

