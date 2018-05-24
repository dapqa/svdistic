// Infer mode of a model.
template <class T>
int infer(T& model, string fname)
{
  // Load testing data.
  ExampleMat X_test(3, model.N_EXAMPLE);
  cout << "Loading testing data..." << endl;
  if (input_pipeline(X_test, model.N_EXAMPLE,
                     "data/corpus/" + fname) == 1)
    return 1;
  cout << "Testing data loaded." << endl;

  // Load model weights.
  cout << "Loading weights..." << endl;
  model.load_weights();
  cout << "Loaded weights." << endl;

  // Predict values on testing data.
  cout << "Starting inference..." << endl;
  ExampleVec preds(model.N_EXAMPLE);
  model.infer(preds, X_test);
  cout << "Inference completed." << endl;

  // Save model inferences.
  cout << "Saving inferences..." << endl;
  save_matrix(preds, "data/saves/" + model.model_id + "-inferred.txt");
  cout << "Inferences saved." << endl;

  return 0;
}


// Score model.
template <class T>
int score(T& model, string fname)
{
  // Load testing data.
  ExampleMat X_val(3, model.N_EXAMPLE);
  cout << "Loading validation data..." << endl;
  if (input_pipeline(X_val, model.N_EXAMPLE,
                     "data/corpus/" + fname) == 1)
    return 1;
  cout << "Validation data loaded." << endl;

  // Load model weights.
  cout << "Loading weights..." << endl;
  model.load_weights();
  cout << "Loaded weights." << endl;

  // Score model
  cout << "Scoring..." << endl;
  float rmse = model.score(X_val);
  cout << "RMSE: " << rmse << endl;
  save_float(rmse, "data/saves/" + model.model_id + "-rmse.txt");
  cout << "Scored." << endl;

  return 0;
}


// Train model.
template <class T>
int train(T& model, string fname)
{
  // Load training data.
  ExampleMat X_tr(3, model.N_EXAMPLE);
  cout << "Loading training data..." << endl;
  if (input_pipeline(X_tr, model.N_EXAMPLE,
                     "data/corpus/" + fname) == 1)
    return 1;
  cout << "Training data loaded." << endl;

  // Initialize model weights.
  cout << "Initializing weights..." << endl;
  model.init_weights();
  cout << "Initialized weights." << endl;

  // Train model.
  cout << "Starting training..." << endl;
  model.train(X_tr);
  cout << "Training completed." << endl;

  cout << "Saving weights..." << endl;
  model.save_weights();
  cout << "Weights saved." << endl;

  return 0;
}

