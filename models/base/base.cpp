#include "./base.h"

/*******************************************************************
 * Higher level model logic.
 ******************************************************************/

// Public function to train model on a set of
// examples for a given number of epochs.
void Base::train(ExampleMat& X_tr)
{
  // float best_score = -1;
  // float current_score = -1;
  cout << "Beginning training for " << N_EPOCHS << " epochs on model "
       << model_id << "." << endl;
  calc_mu(X_tr);

  for (int epc = 0; epc < N_EPOCHS; ++epc)
  {
    for (int ij = 0; ij < N_EXAMPLE; ++ij)
    {
      // Note example count per 10M.
      if (ij % 10000000 == 0)
        cout << "Computing example " << ij << "." << endl;

      // Update for examples
      update(X_tr, ij);

    }

    /*
    current_score = score(X_tr);
    if ((best_score == (-1)) || (current_score < best_score))
    {
      best_score = current_score;
      cout << "Saving best rmse to date: " << best_score << endl;
      cout << "Saving weights..." << endl;
      save_weights();
      cout << "Weights saved." << endl;
    }
    */

    // If it's time to report.
    if ((epc % REPORT_FREQ) == 0)
    {
      cout << "Epoch " << epc << ". RMSE on training set is "
           << score(X_tr) << "." << endl;
    }
    
    // Decay learning rate
    LR *= LR_DECAY;
    cout << "LR: " << LR << endl;
  }

  cout << "Saving weights..." << endl;
  save_weights();
  cout << "Weights saved." << endl;
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
  cout << "RMSE: " << sqrt(all_diffs.squaredNorm() / ((float) N_EXAMPLE)) << endl;

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
  b_u(user) = b_u(user) + LR * (err - REG_B * b_u(user));
}


// Update product bias: b_p = b_p + lr*(e - reg * b_p)
// Calculate for a specific example.
void Base::product_bias(float err, ExampleMat& X, int ij)
{
  const int product = X(1, ij);
  b_p(product) = b_p(product) + LR * (err - REG_B * b_p(product));
}

