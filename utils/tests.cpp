#include "./tests.h"


int main()
{
  srand(0);

  // Starting testing
  cout << "Starting testing..." << endl;
  test_input_pipeline();
  test_load_weights();
  test_dump_weights();
  test_dump_weights_two();
  test_dump_infers();
  cout << "All unit tests completed." << endl;

  return 0;
}


void test_input_pipeline()
{
  cout << endl << "###########################################################"
       << endl << "Running input_pipeline tests..." << endl << endl;
  
  // Actual matrix
  ExampleMat X_tr;
  cout << "Test 1: input_pipeline..." << endl;
  assert(input_pipeline(X_tr, "data/good_test.txt") == 0);

  // Wrong matrix
  ExampleMat Y_tr;
  cout << "Test 2: input_pipeline..." << endl;
  assert(input_pipeline(Y_tr, "data/bad_test.txt") == 1);

  // Validation matrix
  MatrixXi X_tr_val = X_tr_val.Constant(N_EXAMPLE, 4, -5);
  X_tr_val << 0, 0, 1, 0,
              0, 1, 2, 0,
              0, 2, 3, 1,
              1, 0, 4, 0,
              1, 1, 5, 0,
              1, 2, 6, 1,
              2, 0, 7, 0, 
              2, 1, 8, 0,
              2, 1, 9, 1,
              3, 0, 10, 0,
              3, 1, 11, 1;

  // Assertions
  cout << "Test 3: input_pipeline..." << endl;
  assert(X_tr_val.isApprox(X_tr));
  cout << "Test 4: input_pipeline..." << endl;
  assert(!(X_tr_val.isApprox(X_tr * 2)));

  cout << endl << "All input_pipeline tests successful!"
       << endl << "###########################################################"
       << endl << endl;
}


void test_load_weights()
{
  cout << endl << "###########################################################"
       << endl << "Running load_weight tests..." << endl << endl;
  
  // Actual matrix
  UserMat W_u, Ysum_terms;
  ProductMat W_i, W_p;
  UserVec Ru_terms;
  load_weights(W_u, W_i, W_p, Ru_terms, Ysum_terms,
               "weights/test_W_u.txt", "weights/test_W_i.txt",
               "weights/test_W_p.txt", "weights/test_Ru.txt",
               "weights/test_Ysum.txt");

  // Validation matrix
  MatrixXf W_u_val = W_u_val.Constant(N_LATENT, N_USER, -5.0);
  W_u_val << 0, 2, 4, 6,
             1, 3, 5, 7;

  MatrixXf W_i_val = W_i_val.Constant(N_LATENT, N_PRODUCT, -5.0);
  W_i_val << 0.1, 2.1, 4.1,
             1.1, 3.1, 5.1;

  MatrixXf W_p_val = W_p_val.Constant(N_LATENT, N_PRODUCT, -5.0);
  W_p_val << 0.2, 2.2, 4.2,
             1.2, 3.2, 5.2;

  MatrixXf Ysum_val = Ysum_val.Constant(N_LATENT, N_USER, -5.0);
  Ysum_val << 3, 4, 1, 5,
              2, 6, 5.3, 7;

  VectorXf Ru_val = Ru_val.Constant(N_USER, -5.0);
  Ru_val << 4, 6, 7, 9;

  // Assertions
  cout << "Test 1: load_weights..." << endl;
  assert(W_u_val.isApprox(W_u));
  cout << "Test 2: load_weights..." << endl;
  assert(W_i_val.isApprox(W_i));
  cout << "Test 3: load_weights..." << endl;
  assert(W_p_val.isApprox(W_p));
  cout << "Test 4: load_weights..." << endl;
  assert(Ru_val.isApprox(Ru_terms));
  cout << "Test 5: load_weights..." << endl;
  assert(Ysum_val.isApprox(Ysum_terms));

  cout << endl << "All load_weights tests successful!"
       << endl << "###########################################################"
       << endl << endl;
}


void test_dump_weights()
{
  cout << endl << "###########################################################"
       << endl << "Running dump_weights tests..." << endl << endl;

  // Validation matrix
  MatrixXf W_u_val = W_u_val.Constant(N_LATENT, N_USER, -5.0);
  W_u_val << 0, 2, 4, 6,
             1, 3, 5, 7;

  MatrixXf Ysum_val = Ysum_val.Constant(N_LATENT, N_USER, -5.0);
  Ysum_val << 3, 2, 4, 6,
              1, 5.3, 5, 7;

  MatrixXf W_i_val = W_i_val.Constant(N_LATENT, N_PRODUCT, -5.0);
  W_i_val << 0.1, 2.1, 4.1,
             1.1, 3.1, 5.1;

  MatrixXf W_p_val = W_p_val.Constant(N_LATENT, N_PRODUCT, -5.0);
  W_p_val << 0.2, 2.2, 4.2,
             1.2, 3.2, 5.2;

  UserVec Ru_val = Ru_val.Constant(N_USER, -5.0);
  Ru_val << 4, 6, 7, 9;

  // Save matrix
  save_pipeline(W_u_val, W_i_val, W_p_val, Ru_val, Ysum_val,
                "weights/temp_W_u.txt", "weights/temp_W_i.txt",
                "weights/temp_W_p.txt", "weights/temp_Ru.txt",
                "weights/temp_Ysum.txt");
  
  // Actual matrix
  UserMat W_u, Ysum_terms;
  ProductMat W_i, W_p;
  UserVec Ru_terms;
  load_weights(W_u, W_i, W_p, Ru_terms, Ysum_terms,
                "weights/temp_W_u.txt", "weights/temp_W_i.txt",
                "weights/temp_W_p.txt", "weights/temp_Ru.txt",
                "weights/temp_Ysum.txt");

  // Assertions
  cout << "Test 1: dump_weightss..." << endl;
  assert(W_u_val.isApprox(W_u));
  cout << "Test 2: dump_weightss..." << endl;
  assert(W_i_val.isApprox(W_i));
  cout << "Test 3: dump_weightss..." << endl;
  assert(W_p_val.isApprox(W_p));
  cout << "Test 4: dump_weightss..." << endl;
  assert(Ru_val.isApprox(Ru_terms));
  cout << "Test 5: dump_weightss..." << endl;
  assert(Ysum_val.isApprox(Ysum_terms));

  cout << endl << "All dump_weightss tests successful!"
       << endl << "###########################################################"
       << endl << endl;
}


void test_dump_weights_two()
{
  cout << endl << "###########################################################"
       << endl << "Running dump_weights_two tests..." << endl << endl;

  // Validation matrix
  MatrixXf W_u_val = W_u_val.Constant(N_LATENT, N_USER, -5.0);
  W_u_val << 9, 7, 5, 3,
             8, 6, 4, 2;

  MatrixXf W_i_val = W_i_val.Constant(N_LATENT, N_PRODUCT, -5.0);
  W_i_val << 9.1, 7.1, 5.1,
             8.1, 6.1, 4.1;

  MatrixXf W_p_val = W_p_val.Constant(N_LATENT, N_PRODUCT, -5.0);
  W_p_val << 9.2, 7.2, 5.2,
             8.2, 6.2, 4.2;

  MatrixXf Ysum_val = Ysum_val.Constant(N_LATENT, N_USER, -5.0);
  Ysum_val << 3.342, 9, 9, 6,
              1, 5.3, 9, 7;

  UserVec Ru_val = Ru_val.Constant(N_USER, -5.0);
  Ru_val << 9, 3, 2, 9;

  // Save matrix
  save_pipeline(W_u_val, W_i_val, W_p_val, Ru_val, Ysum_val,
                "weights/temp_W_u.txt", "weights/temp_W_i.txt",
                "weights/temp_W_p.txt", "weights/temp_Ru.txt",
                "weights/temp_Ysum.txt");
  
  // Actual matrix
  UserMat W_u, Ysum_terms;
  ProductMat W_i, W_p;
  UserVec Ru_terms;
  load_weights(W_u, W_i, W_p, Ru_terms, Ysum_terms,
                "weights/temp_W_u.txt", "weights/temp_W_i.txt",
                "weights/temp_W_p.txt", "weights/temp_Ru.txt",
                "weights/temp_Ysum.txt");

  // Assertions
  cout << "Test 1: dump_weightss..." << endl;
  assert(W_u_val.isApprox(W_u));
  cout << "Test 2: dump_weightss..." << endl;
  assert(W_i_val.isApprox(W_i));
  cout << "Test 3: dump_weightss..." << endl;
  assert(W_p_val.isApprox(W_p));
  cout << "Test 4: dump_weightss..." << endl;
  assert(Ru_val.isApprox(Ru_terms));
  cout << "Test 5: dump_weightss..." << endl;
  assert(Ysum_val.isApprox(Ysum_terms));

  cout << endl << "All dump_weights_twos tests successful!"
       << endl << "###########################################################"
       << endl << endl;
}


void test_dump_infers()
{
  cout << endl << "###########################################################"
       << endl << "Running dump_infers tests..." << endl << endl;
  
  // Validation matrix
  ExampleVec Pred_val = ExampleVec::Constant(-5);
  Pred_val << 1, 4, 5, 6, 8, 0, 2, 2, 2, 3, 3;
  // Assertions
  cout << "Test 1: dump_infers..." << endl;
  save_pipeline(Pred_val, "data/good_test_preds.txt");
  cout << endl << "All dump_infers tests successful!"
       << endl << "###########################################################"
       << endl << endl;
}

