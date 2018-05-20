/***********************************************
 * Higher level training steps
 **********************************************/

// Initialize on corpus
void TimeSVDpp::per_corpus(ExampleMat& X)
{
  // Calculate global average
  calc_mu(X);

  // Calculate Ru: number of reviews for each user.
  Ru.setZero();
  for (int ij = 0; ij < N_EXAMPLE; ++ij)
  {
    // Increment count for a user.
    ++Ru(X(ij, 0));
  }
  // Now we make Ru = |Ru|^(-1/2)
  Ru = Ru.cwiseSqrt().cwiseInverse();

  // Initialize for the first epoch and user.
  per_epoch(X);
}

// Initialize for an epoch
void TimeSVDpp::per_epoch(ExampleMat& X)
{
  // Initialize Ysum: the sum of W_i for the products of each user.
  
  // Zero Ysum first.
  Ysum.setZero(N_LATENT, N_USER);
  for (int ij = 0; ij < N_EXAMPLE; ++ij)
  {
    // Update ysum for the user.
    Ysum.col(X(ij, 0)) += W_i.col(X(ij, 1));
  }

  // Initialize for the first user.
  // We need to init products seen to handle stuff
  products_seen = ProductVec::Constant(-1);
  products_count = 0;
  per_user();
}

// Initalize user info
void TimeSVDpp::per_user()
{
  // We handle the previos user
  // Note this is skipped if first user.
  for (int k = 0; k < N_PRODUCT; ++k)
  {
    // Quit once we've stopped seeing product.
    if (products_seen(k) == -1)
    {
      break;
    }
    // products_seen(k) = j
    implicit_weight(products_seen(k));
  }

  // We zero the implicit term sum for this user.
  implicit_terms.setZero();
  // We forget all the products we've seen.
  products_seen = ProductVec::Constant(-1);
  products_count = 0;
}

// Initialize for an element.
void TimeSVDpp::update(ExampleMat& X, int ij)
{
  float err = pred_diff(X, ij);
  // Update standard SVD rules.
  user_bias(err, X, ij);
  product_bias(err, X, ij);
  product_weight(err, X, ij);
  user_weight(err, X, ij);
  // Update implicit_terms.
  accum_implicit(err, X, ij);
  // Remember that we've seen this product.
  products_seen(products_count) = X(ij, 1);
  ++products_count;
}

/***********************************************
 * Predict stuff
 **********************************************/
// pred = mu + b_u + b_p + W_p(W_u + |R(u)|^-1/2 sum y_j)
// Calculate for a specific example.
float TimeSVDpp::predict(ExampleMat& X, int ij)
{
  const int i = X(ij, 0);
  const int j = X(ij, 1);
  return mu + b_p(j) + b_u(i) + W_p.col(j).transpose() *
      (W_u.col(i) + Ru(i) * Ysum.col(i));
}


void TimeSVDpp::product_time_bias(float err, ExampleMat& X, int ij)
{
}

void TimeSVDpp::user_time_bias(float err, ExampleMat& X, int ij)
{
}

void TimeSVDpp::user_time_weight(float err, ExampleMat& X, int ij)
{
}

