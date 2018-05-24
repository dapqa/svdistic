/***********************************************
 * Higher level training steps
 **********************************************/

// Initialize on corpus
void SVD::per_corpus(ExampleMat& X)
{
  // Calculate global average
  calc_mu(X);
}

// Initialize for an epoch
void SVD::per_epoch(ExampleMat& X)
{
}

// Initalize user info
void SVD::per_user(ExampleMat& X, int ij)
{
}

// Initialize for an element.
void SVD::update(ExampleMat& X, int ij)
{
  float err = pred_diff(X, ij);
  user_bias(err, X, ij);
  product_bias(err, X, ij);
  user_weight(err, X, ij);
  product_weight(err, X, ij);
}


/***********************************************
 * Saving/loading...
 **********************************************/

// Initialize all weights.
void SVD::init_weights()
{
  W_u.setRandom(N_LATENT, N_USER);
  W_p.setRandom(N_LATENT, N_PRODUCT);
  Base::init_weights();
}

// Save weights into file.
void SVD::save_weights()
{
  save_matrix<float, N_LATENT, -1>(W_u, "data/saves/" + model_id + "-W_u");
  save_matrix<float, N_LATENT, -1>(W_p, "data/saves/" + model_id + "-W_p");
  Base::save_weights();
}

// Load weights from file.
void SVD::load_weights()
{
  load_matrix<float, N_LATENT, -1>(W_u, "data/saves/" + model_id + "-W_u", N_LATENT, N_USER);
  load_matrix<float, N_LATENT, -1>(W_p, "data/saves/" + model_id + "-W_p", N_LATENT, N_PRODUCT);
  Base::load_weights();
}

