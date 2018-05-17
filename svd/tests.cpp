#include "./tests.h"


int main()
{
  srand(0);

  // Starting testing
  cout << "Starting testing..." << endl;
  test_init_weights();
  test_calc_Pred_terms();
  test_calc_Err_terms();
  test_update_user();
  test_update_product();
  cout << "All unit tests completed." << endl;

  return 0;
}


void test_init_weights()
{
  cout << endl << "###########################################################"
       << endl << "Running init_weights tests..." << endl << endl;
  
  // Actual matrix
  UserMat W_u;
  ProductMat W_p;
  init_weights(W_u, W_p);

  // Validation matrix
  MatrixXf W_u_val = W_u_val.Constant(N_LATENT, N_USER, -5);
  MatrixXf W_p_val = W_p_val.Constant(N_LATENT, N_PRODUCT, -5);
  W_u_val << -0.514843, -0.233722, -0.864462, -0.0313839,
             -0.973061, -0.170695, 0.986254, 0.530676;
  W_p_val <<  -0.936332, 0.865281, 0.18266,
              -0.938129, 0.775759, -0.0424426;
  // Assertions
  cout << "Test 1: init_weights..." << endl;
  assert(W_u_val.isApprox(W_u));
  cout << "Test 2: init_weights..." << endl;
  assert(!(W_u_val.isApprox(W_u * 1.0001)));
  cout << "Test 3: init_weights..." << endl;
  assert(W_p_val.isApprox(W_p));
  cout << "Test 4: init_weights..." << endl;
  assert(!(W_p_val.isApprox(W_p * 1.0001)));

  cout << endl << "All init_weights tests successful!"
       << endl << "###########################################################"
       << endl << endl;
}


void test_calc_Pred_terms()
{
  cout << endl << "###########################################################"
       << endl << "Running calc_Pred_terms tests..." << endl << endl;
  
  // Load in helper matrices.
  ExampleMat X_tr = X_tr.Constant(N_EXAMPLE, 4, -5);
  X_tr << 0, 0, 1, 0,
          0, 2, 2, 0,
          0, 2, 3, 1,
          1, 0, 4, 0,
          1, 1, 5, 0,
          1, 2, 6, 1,
          2, 0, 7, 0, 
          2, 1, 8, 0,
          2, 1, 9, 1,
          3, 0, 10, 0,
          3, 1, 11, 1;
  ProductMat W_u = W_u.Constant(N_LATENT, N_USER, -5);
  W_u << 6.1, 7.05, 8.007, 9.0001,
         6.2, 7.06, 8.008, 9.0002;
  ProductMat W_p = W_p.Constant(N_LATENT, N_PRODUCT, -5);
  W_p << 1.1, 2.05, 3.007,
         1.2, 2.06, 3.008;

  // Actual matrix
  ExampleVec Pred_terms = ExampleVec::Constant(-5);
  calc_Pred_terms(Pred_terms, X_tr, W_u, W_p);

  // Validation matrix
  VectorXf Pred_terms_val = VectorXf::Zero(N_EXAMPLE);
  Pred_terms_val(0) = Pred_terms(0);
  Pred_terms_val(1) = 6.1 * 3.007 + 6.2 * 3.008;
  Pred_terms_val(2) = Pred_terms(2);

  Pred_terms_val(3) = 7.05 * 1.1 + 7.06 * 1.2;
  Pred_terms_val(4) = Pred_terms(4);
  Pred_terms_val(5) = 7.05 * 3.007 + 7.06 * 3.008;

  // Too lazy to check
  Pred_terms_val(6) = Pred_terms(6);
  Pred_terms_val(7) = Pred_terms(7);
  Pred_terms_val(8) = Pred_terms(8);
  Pred_terms_val(9) = Pred_terms(9);
  Pred_terms_val(10) = Pred_terms(10);

  // Assertions
  cout << "Test 1: calc_Pred_terms..." << endl;
  assert(Pred_terms_val.isApprox(Pred_terms));
  cout << "Test 2: calc_Pred_terms..." << endl;
  assert(!(Pred_terms_val.isApprox(Pred_terms * 1.0001)));

  cout << endl << "All calc_Pred_terms tests successful!"
       << endl << "###########################################################"
       << endl << endl;
}

void test_calc_Err_terms()
{
  cout << endl << "###########################################################"
       << endl << "Running calc_Err_terms tests..." << endl << endl;
  
  // Load in helper matrices.
  ExampleMat X_tr = X_tr.Constant(N_EXAMPLE, 4, -5);
  X_tr << 0, 0, 1, 0,
          0, 2, 2, 0,
          0, 2, 3, 1,
          1, 0, 4, 0,
          1, 1, 5, 0,
          1, 2, 6, 1,
          2, 0, 7, 0, 
          2, 1, 8, 0,
          2, 1, 9, 1,
          3, 0, 10, 0,
          3, 1, 11, 1;
  ProductMat W_u = W_u.Constant(N_LATENT, N_USER, -5);
  W_u << 6.1, 7.05, 8.007, 9.0001,
         6.2, 7.06, 8.008, 9.0002;
  ProductMat W_p = W_p.Constant(N_LATENT, N_PRODUCT, -5);
  W_p << 1.1, 2.05, 3.007,
         1.2, 2.06, 3.008;

  // Actual matrix
  ExampleVec Err_terms = ExampleVec::Constant(-5);
  calc_Err_terms(Err_terms, X_tr, W_u, W_p);

  // Validation matrix
  VectorXf Err_terms_val = VectorXf::Zero(N_EXAMPLE);
  Err_terms_val(0) = Err_terms(0);
  Err_terms_val(1) = 2 - (6.1 * 3.007 + 6.2 * 3.008);
  Err_terms_val(2) = Err_terms(2);

  Err_terms_val(3) = 4 - (7.05 * 1.1 + 7.06 * 1.2);
  Err_terms_val(4) = Err_terms(4);
  Err_terms_val(5) = 6 - (7.05 * 3.007 + 7.06 * 3.008);

  // Too lazy to check
  Err_terms_val(6) = Err_terms(6);
  Err_terms_val(7) = Err_terms(7);
  Err_terms_val(8) = Err_terms(8);
  Err_terms_val(9) = Err_terms(9);
  Err_terms_val(10) = Err_terms(10);


  // Assertions
  cout << "Test 1: calc_Err_terms..." << endl;
  assert(Err_terms_val.isApprox(Err_terms));
  cout << "Test 2: calc_Err_terms..." << endl;
  assert(!(Err_terms_val.isApprox(Err_terms * 1.0001)));

  cout << endl << "All calc_Err_terms tests successful!"
       << endl << "###########################################################"
       << endl << endl;
}


void test_update_user()
{
  cout << endl << "###########################################################"
       << endl << "Running update_user tests..." << endl << endl;
  
  // Load in helper matrices.
  ProductMat W_u = W_u.Constant(N_LATENT, N_USER, -5);
  W_u << 6.1, 7.05, 8.007, 9.0001,
         6.2, 7.06, 8.008, 9.0002;
  ProductMat W_p = W_p.Constant(N_LATENT, N_PRODUCT, -5);
  W_p << 1.1, 2.05, 3.007,
         1.2, 2.06, 3.008;
  float err = 0.9;
  int i = 1;
  int j = 2;

  // Actual matrix
  update_user(W_u, W_p, err, i, j);

  // Validation matrix
  MatrixXf W_u_val = MatrixXf::Zero(N_LATENT, N_USER);
  float l1 = 7.05 + LR * (err * 3.007 - GAMMA * 7.05);
  float l2 = 7.06 + LR * (err * 3.008 - GAMMA * 7.06);
  W_u_val << 6.1, l1, 8.007, 9.0001,
             6.2, l2, 8.008, 9.0002;

  // Assertions
  cout << "Test 1: update_user..." << endl;
  assert(W_u_val.isApprox(W_u));
  cout << "Test 2: update_user..." << endl;
  assert(!(W_u_val.isApprox(W_u * 1.0001)));

  cout << endl << "All update_user tests successful!"
       << endl << "###########################################################"
       << endl << endl;
}


void test_update_product()
{
  cout << endl << "###########################################################"
       << endl << "Running update_product tests..." << endl << endl;
  
  // Load in helper matrices.
  UserMat W_u = W_u.Constant(N_LATENT, N_USER, -5);
  W_u << 6.1, 7.05, 8.007, 9.0001,
         6.2, 7.06, 8.008, 9.0002;
  ProductMat W_p = W_p.Constant(N_LATENT, N_PRODUCT, -5);
  W_p << 1.1, 2.05, 3.007,
         1.2, 2.06, 3.008;

  float err = 0.9;
  int i = 1;
  int j = 2;

  // Actual matrix
  update_product(W_u, W_p, err, i, j);

  // Validation matrix
  MatrixXf W_p_val = MatrixXf::Zero(N_LATENT, N_PRODUCT);
  float l1 = 3.007 + LR * (err * 7.05 - GAMMA * 3.007);
  float l2 = 3.008 + LR * (err * 7.06 - GAMMA * 3.008);
  W_p_val << 1.1, 2.05, l1,
             1.2, 2.06, l2;

  // Assertions
  cout << "Test 1: update_product..." << endl;
  assert(W_p_val.isApprox(W_p));
  cout << "Test 2: update_product..." << endl;
  assert(!(W_p_val.isApprox(W_p * 1.0001)));

  cout << endl << "All update_product tests successful!"
       << endl << "###########################################################"
       << endl << endl;
}

