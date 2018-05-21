/***********************************************
 * Higher level training steps
 **********************************************/

// Initialize on corpus
void SVDpp::per_corpus(ExampleMat& X)
{
  // Calculate global average
  calc_mu(X);

  // Calculate Ru: number of reviews for each user.
  Ru.setZero(N_USER);
  for (int ij = 0; ij < N_EXAMPLE; ++ij)
  {
    // Increment count for a user.
    ++Ru(X(0, ij));
  }
  // Now we make Ru = |Ru|^(-1/2)
  Ru = Ru.cwiseSqrt().cwiseInverse();

  // Initialize for the first epoch.
  per_epoch(X);
}

// Initialize for an epoch
void SVDpp::per_epoch(ExampleMat& X)
{
  // Initialize Ysum: the sum of W_i for the products of each user.
  
  // Zero Ysum first.
  Ysum.setZero(N_LATENT, N_USER);
  for (int ij = 0; ij < N_EXAMPLE; ++ij)
  {
    // Update ysum for the user.
    Ysum.col(X(0, ij)) += W_i.col(X(1, ij));
  }

  // Initialize for the first user.
  user_start_ij = 0;
}

// Initalize user info
void SVDpp::per_user(ExampleMat& X, int ij)
{
  // We handle the previos user
  // Note this is skipped if first user.
  for (int k = user_start_ij; k <= ij; ++k)
  {
    implicit_weight(X(1, k));
  }

  implicit_terms.setZero();
  user_start_ij = ij + 1;
}

// Initialize for an element.
void SVDpp::update(ExampleMat& X, int ij)
{
  float err = pred_diff(X, ij);
  // Update standard SVD rules.
  user_bias(err, X, ij);
  product_bias(err, X, ij);
  product_weight(err, X, ij);
  user_weight(err, X, ij);
  // Update implicit_terms.
  accum_implicit(err, X, ij);
}

/***********************************************
 * Predict stuff
 **********************************************/
// pred = mu + b_u + b_p + W_p(W_u + |R(u)|^-1/2 sum y_j)
// Calculate for a specific example.
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
// Calculate for a specific example.
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

