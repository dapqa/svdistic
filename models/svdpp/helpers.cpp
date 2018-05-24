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
  user_bias(err, X, ij);
  product_bias(err, X, ij);
  product_weight(err, X, ij);
  user_weight(err, X, ij);
  accum_implicit(err, X, ij);
}


/***********************************************
 * Saving/loading...
 **********************************************/

// Initialize all weights.
void SVDpp::init_weights()
{
  implicit_terms.setZero();
  W_i.setRandom(N_LATENT, N_PRODUCT);
  SVD::init_weights();
}

// Save weights into file.
void SVDpp::save_weights()
{
  save_matrix<float, N_LATENT, -1>(W_i, "data/saves/" + model_id + "-W_i");
  save_matrix<float, N_LATENT, -1>(Ysum, "data/saves/" + model_id + "-Ysum");
  save_matrix<float, -1, 1>(Ru, "data/saves/" + model_id + "-Ru");
  SVD::save_weights();
}

// Load weights from file.
void SVDpp::load_weights()
{
  load_matrix<float, N_LATENT, -1>(W_i, "data/saves/" + model_id + "-W_i", N_LATENT, N_PRODUCT);
  load_matrix<float, N_LATENT, -1>(Ysum, "data/saves/" + model_id + "-Ysum", N_LATENT, N_USER);
  load_matrix<float, -1, 1>(Ru, "data/saves/" + model_id + "-Ru", N_USER, 1);
  SVD::load_weights();
}

