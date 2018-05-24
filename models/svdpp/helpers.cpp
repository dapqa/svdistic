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
    ++Ru(X(0, ij));
  }

  // Now we make Ru = |Ru|^(-1/2)
  RuNorm = Ru.cwiseAbs().cwiseSqrt().cwiseInverse();
}

// Initialize for an epoch
void SVDpp::per_epoch(ExampleMat& X)
{
}

// Initalize user info
void SVDpp::init_user(ExampleMat& X, int ij)
{
  // Note the start of this user.
  user_start_ij = ij;

  // Build initial Ysum calculation.
  const int i = X(0, ij);
  Ysum.col(i).setZero();
  for (int ijj = user_start_ij; ijj < user_start_ij + Ru(i); ++ijj)
  {
    Ysum.col(i) += W_i.col(X(1, ijj));
  }

  // Clear the implicit term which will accumulate.
  implicit_term.setZero();
  scale = 0;
}

// End user
void SVDpp::end_user(ExampleMat& X, int ij)
{
  if (scale != 0)
  {
    // Update the weight of all products.
    implicit_term = implicit_term / (float) scale;
    implicit_weight(X, X(0, ij));
  }
}

// Initialize for an element.
void SVDpp::update(ExampleMat& X, int ij)
{
  float err = pred_diff(X, ij); 
  user_bias(err, X, ij);
  product_bias(err, X, ij);
  product_weight(err, X, ij);
  user_weight(err, X, ij);
 
  // Manipualtes implicit term
  accum_implicit(err, X, ij);
  scale += 1;

  // Occasional update every 500 products.
  if (((ij - user_start_ij) + 1) % 500 == 0)
  {
    const int i = X(0, ij);

    implicit_term = implicit_term / (float) scale;
    implicit_weight(X, i);
    implicit_term.setZero();
    scale = 0;

    Ysum.col(i).setZero();
    for (int ijj = user_start_ij; ijj < user_start_ij + Ru(i); ++ijj)
    {
      Ysum.col(i) += W_i.col(X(1, ijj));
    }
  }
}


/***********************************************
 * Saving/loading...
 **********************************************/

// Initialize all weights.
void SVDpp::init_weights()
{
  W_i.setRandom(N_LATENT, N_PRODUCT);
  W_i = 0.00000001 * (W_i + W_i.Ones(N_LATENT, N_PRODUCT));
  Ysum.setZero(N_LATENT, N_USER);
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
  RuNorm = Ru.cwiseAbs().cwiseSqrt().cwiseInverse();
  SVD::load_weights();
}

