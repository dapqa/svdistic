#include "./base.h"

/*******************************************************************
 * Higher level model logic.
 ******************************************************************/

// Public function to train model on a set of
// examples for a given number of epochs.
void Base::train(ExampleMat& X_tr)
{
  cout << "Beginning training for " << N_EPOCHS << " epochs on model "
       << model_id << "." << endl;
  calc_mu(X_tr);

  for (int epc = 0; epc < N_EPOCHS; ++epc)
  {
    for (int ij = 0; ij < N_EXAMPLE; ++ij)
    {
      // Update for examples
      update(X_tr, ij);

    }
    // If it's time to report.
    if ((epc % REPORT_FREQ) == 0)
    {
      cout << "Epoch " << epc << ". RMSE on training set is "
           << score(X_tr) << "." << endl;
    }
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
  cout << "RMSE: " << all_diffs.squaredNorm() << endl;

  return sqrt(all_diffs.squaredNorm() / ((float) N_EXAMPLE));
}


// Calculate differencce between expected and real
// user ratings for a specific item.
// diff_ij = y_ij - p_ij.
float Base::pred_diff(ExampleMat& X, int ij)
{
  return X(2, ij) - predict(X, ij);
}


/*******************************************************************
 * Update bias terms.
 ******************************************************************/

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
  b_u(user) = b_u(user) + LR * (err - REG * b_u(user));
}


// Update product bias: b_p = b_p + lr*(e - reg * b_p)
// Calculate for a specific example.
void Base::product_bias(float err, ExampleMat& X, int ij)
{
  const int product = X(1, ij);
  b_p(product) = b_p(product) + LR * (err - REG * b_p(product));
}

