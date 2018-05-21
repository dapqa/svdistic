#include "./model.h"
#include "./helpers.cpp"


void SVDpp::train(ExampleMat& X_tr)
{
  float best_score = -1;
  float current_score = -1;
  cout << "Beginning training for " << N_EPOCHS << " epochs on model "
       << model_id << "." << endl;
  per_corpus(X_tr);

  for (int epc = 0; epc < N_EPOCHS; ++epc)
  {
    per_epoch(X_tr);
    for (int ij = 0; ij < N_EXAMPLE; ++ij)
    {
      // Note example count per 10M.
      if (ij % 10000000 == 0)
        cout << "Computing example " << ij << "." << endl;

      update(X_tr, ij);
      // If this is the final entry of the user.
      if ((ij + 1 == N_EXAMPLE) || (X_tr(0, ij) != X_tr(0, ij + 1)))
      {
        per_user(X_tr, ij);
      }
    }

    current_score = score(X_tr);
    if ((best_score == (-1)) || (current_score < best_score))
    {
      best_score = current_score;
      cout << "Saving best rmse to date: " << best_score << endl;
      cout << "Saving weights..." << endl;
      save_weights();
      cout << "Weights saved." << endl;
    }

    // If it's time to report.
    if ((epc % REPORT_FREQ) == 0)
    {
      cout << "Epoch " << epc << " finished. RMSE on training set is "
           << Base::score(X_tr) << "." << endl;
    }
  }
  cout << "Training is complete." << endl;
}


// Public function to obtain inferences on examples.
// The preds matrix is mutated to contain the new
// inferences.
void SVDpp::infer(ExampleVec& preds, ExampleMat& X_test)
{
  Base::infer(preds, X_test);
}


// Public function to score model on examples.
// RMSE is returned.
float SVDpp::score(ExampleMat& X_val)
{
  return Base::score(X_val);
}


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

