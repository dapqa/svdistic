#include "./base.h"


/***********************************************
 * Higher level training step.
 **********************************************/

// Public function to train model on a set of
void Base::train(ExampleMat& X_tr)
{
  cout << "Beginning training for " << N_EPOCHS << " epochs on model "
       << model_id << "." << endl;
  per_corpus(X_tr);
  for (int epc = 0; epc < N_EPOCHS; ++epc)
  {
    per_epoch(X_tr);
    for (int ij = 0; ij < N_EXAMPLE; ++ij)
    {
      if ((ij == 0) || (X_tr(0, ij) != X_tr(0, ij - 1)))
      {
        init_user(X_tr, ij);
      }
      update(X_tr, ij);
      if ((ij == N_EXAMPLE - 1) || (X_tr(0, ij) != X_tr(0, ij + 1)))
      {
        end_user(X_tr, ij);
      }
    }
    if ((epc % REPORT_FREQ) == 0)
    {
      cout << "Epoch " << epc + 1 << " finished." << endl;
      cout << "RMSE: " << Base::score(X_tr) << endl;
    }
    LR *= LR_DECAY;
  }

  cout << "Training is complete." << endl;
}


// Public function to obtain inferences on examples.
void Base::infer(ExampleVec& preds, ExampleMat& X_test)
{
  for (int ij = 0; ij < N_EXAMPLE; ++ij)
  {
    preds(ij) = predict(X_test, ij);
  }
}


// Public function to score model on examples.
// rmse = |x_1, x_2... x_k|^2
float Base::score(ExampleMat& X_val)
{
  ExampleVec all_diffs(N_EXAMPLE);
  for (int ij = 0; ij < N_EXAMPLE; ++ij)
  {
    all_diffs(ij) = pred_diff(X_val, ij);
  }

  return sqrt(all_diffs.squaredNorm() / ((float) N_EXAMPLE));
}


// Calculate differencce between expected and real
// user ratings for a specific item.
// diff_ij = y_ij - p_ij.
float Base::pred_diff(ExampleMat& X, int ij)
{
  return X(2, ij) - predict(X, ij);
}


/***********************************************
 * Update weights
 **********************************************/

// Calculate global average: mu.
// Calculate for all examples.
void Base::calc_mu(ExampleMat& X)
{
  mu = 0;
  for (int ij = 0; ij < N_EXAMPLE; ++ij)
  {
    mu += X(2, ij);
  }
  mu = mu / N_EXAMPLE;
}


// Update user bias: b_u = b_u + lr*(e - reg * b_u)
// Calculate for a specific example.
void Base::user_bias(float err, ExampleMat& X, int ij)
{
  const int user = X(0, ij);
  b_u(user) = b_u(user) + LR * (err - REG_B * b_u(user));
}


// Update product bias: b_p = b_p + lr*(e - reg * b_p)
// Calculate for a specific example.
void Base::product_bias(float err, ExampleMat& X, int ij)
{
  const int product = X(1, ij);
  b_p(product) = b_p(product) + LR * (err - REG_B * b_p(product));
}


/***********************************************
 * Init, save, load weights
 **********************************************/

// Initialize all weights.
void Base::init_weights()
{
  b_u.setZero(N_USER);
  b_p.setZero(N_PRODUCT);
}

// Save weights into file.
void Base::save_weights()
{
  save_float(mu, "data/saves/" + model_id + "-mu");
  save_matrix<float, -1, 1>(b_u, "data/saves/" + model_id + "-b_u");
  save_matrix<float, -1, 1>(b_p, "data/saves/" + model_id + "-b_p");
}

// Load weights from file.
void Base::load_weights()
{
  mu = load_float("data/saves/" + model_id + "-mu");
  load_matrix<float, -1, 1>(b_u, "data/saves/" + model_id + "-b_u", N_USER, 1);
  load_matrix<float, -1, 1>(b_p, "data/saves/" + model_id + "-b_p", N_PRODUCT, 1);
}

