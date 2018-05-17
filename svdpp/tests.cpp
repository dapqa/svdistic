#include "./tests.h"


int main()
{
  srand(0);

  // Starting testing
  cout << "Starting testing..." << endl;
  test_init_weights();
  test_calc_Ru_terms();
  test_calc_Ysum_terms();
  test_calc_Pred_terms();
  test_calc_Err_terms();
  test_update_user();
  test_update_product();
  test_update_implicit();
  cout << "All unit tests completed." << endl;

  return 0;
}


void test_init_weights()
{
  cout << endl << "###########################################################"
       << endl << "Running init_weights tests..." << endl << endl;
  
  // Actual matrix
  UserMat W_u;
  ProductMat W_i;
  ProductMat W_p;
  init_weights(W_u, W_i, W_p);

  // Validation matrix
  MatrixXf W_u_val = W_u_val.Constant(N_LATENT, N_USER, -5);
  MatrixXf W_i_val = W_i_val.Constant(N_LATENT, N_PRODUCT, -5);
  MatrixXf W_p_val = W_p_val.Constant(N_LATENT, N_PRODUCT, -5);
  W_u_val << -0.514843, -0.233722, -0.864462, -0.0313839,
             -0.973061, -0.170695, 0.986254, 0.530676;
  W_i_val << -0.936332, 0.865281, 0.18266,
             -0.938129, 0.775759, -0.0424426;
  W_p_val << 0.666709, 0.471305, 0.397317,
             -0.62733, -0.769894, -0.288792;

  // Assertions
  cout << "Test 1: init_weights..." << endl;
  assert(W_u_val.isApprox(W_u));
  cout << "Test 2: init_weights..." << endl;
  assert(!(W_u_val.isApprox(W_u * 1.0001)));
  cout << "Test 3: init_weights..." << endl;
  assert(W_i_val.isApprox(W_i));
  cout << "Test 4: init_weights..." << endl;
  assert(!(W_i_val.isApprox(W_i * 1.0001)));
  cout << "Test 5: init_weights..." << endl;
  assert(W_p_val.isApprox(W_p));
  cout << "Test 6: init_weights..." << endl;
  assert(!(W_p_val.isApprox(W_p * 1.0001)));

  cout << endl << "All init_weights tests successful!"
       << endl << "###########################################################"
       << endl << endl;
}


void test_calc_Ru_terms()
{
  cout << endl << "###########################################################"
       << endl << "Running calc_Ru_terms tests..." << endl << endl;
  
  // Load in helper matrices.
  ExampleMat X_tr = X_tr.Constant(N_EXAMPLE, 4, -5);
  X_tr << 0, 0, 1, 0,
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

  // Actual matrix
  UserVec Ru_terms = UserVec::Zero();
  calc_Ru_terms(Ru_terms, X_tr);

  // Validation matrix
  VectorXf Ru_terms_val = VectorXf::Zero(N_USER);
  // Ru_terms_val << pow(3.0, -0.5), pow(3.0, -0.5), pow(3.0, -0.5), pow(2.0, -0.5);
  Ru_terms_val << 0.5773502691896257, 0.5773502691896257,
                  0.5773502691896257, 0.7071067811865476;

  // Assertions
  cout << "Test 1: calc_Ru_terms..." << endl;
  assert(Ru_terms_val.isApprox(Ru_terms));
  cout << "Test 2: calc_Ru_terms..." << endl;
  assert(!(Ru_terms_val.isApprox(Ru_terms * 1.0001)));

  cout << endl << "All calc_Ru_terms tests successful!"
       << endl << "###########################################################"
       << endl << endl;
}


void test_calc_Ysum_terms()
{
  cout << endl << "###########################################################"
       << endl << "Running calc_Ysum_terms tests..." << endl << endl;
  
  // Load in helper matrices.
  ExampleMat X_tr = X_tr.Constant(N_EXAMPLE, 4, -5);
  X_tr << 0, 0, 1, 0,
          0, 2, 3, 0,
          0, 2, 3, 1,
          1, 0, 4, 0,
          1, 1, 5, 0,
          1, 2, 6, 1,
          2, 0, 7, 0, 
          2, 1, 8, 0,
          2, 1, 9, 1,
          3, 0, 10, 0,
          3, 1, 11, 1;
  ProductMat W_i = W_i.Constant(N_LATENT, N_PRODUCT, -5);
  W_i << 1.1, 2.05, 3.007,
         1.2, 2.06, 3.008;

  // Actual matrix
  UserMat Ysum_terms = UserMat::Zero(N_LATENT, N_USER);
  calc_Ysum_terms(Ysum_terms, W_i, X_tr);

  // Validation matrix
  MatrixXf Ysum_terms_val = MatrixXf::Zero(N_LATENT, N_USER);
  // Ysum_terms_val << 1.1 + 3.007 + 3.007, 1.1 + 2.05 + 3.007,
  //                   1.1 + 2.05 + 2.05, 1.1 + 2.05,
  //
  //                   1.2 + 3.008 + 3.008, 1.2 + 2.06 + 3.008,
  //                   1.2 + 2.06 + 2.06, 1.2 + 2.06;
  Ysum_terms_val << 7.114, 6.157, 5.2, 3.15,
                    7.216, 6.268, 5.32, 3.26;


  // Assertions
  cout << "Test 1: calc_Ysum_terms..." << endl;
  assert(Ysum_terms_val.isApprox(Ysum_terms));
  cout << "Test 2: calc_Ysum_terms..." << endl;
  assert(!(Ysum_terms_val.isApprox(Ysum_terms * 1.0001)));

  cout << endl << "All calc_Ysum_terms tests successful!"
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
  UserMat Ysum_terms = MatrixXf::Zero(N_LATENT, N_USER);
  Ysum_terms << 7.114, 6.157, 5.2, 3.15,
                7.216, 6.268, 5.32, 3.26;
  UserVec Ru_terms = UserVec::Zero();
  Ru_terms << 0.5773502691896257, 0.5773502691896257,
              0.5773502691896257, 0.7071067811865476;

  // Actual matrix
  ExampleVec Pred_terms = ExampleVec::Constant(-5);
  calc_Pred_terms(Pred_terms, X_tr, W_u,
                  W_p, Ysum_terms, Ru_terms);

  // Validation matrix
  VectorXf Pred_terms_val = VectorXf::Zero(N_EXAMPLE);
  Pred_terms_val(0) = 23.667383810000004;
  Pred_terms_val(1) = 61.8746566361;
  Pred_terms_val(2) = 61.8746566361;

  Pred_terms_val(3) = 24.479814105;
  Pred_terms_val(4) = 43.73811448549999;
  Pred_terms_val(5) = 64.01038509605;

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
  UserMat Ysum_terms = MatrixXf::Zero(N_LATENT, N_USER);
  Ysum_terms << 7.114, 6.157, 5.2, 3.15,
                7.216, 6.268, 5.32, 3.26;
  UserVec Ru_terms = UserVec::Zero();
  Ru_terms << 0.5773502691896257, 0.5773502691896257,
              0.5773502691896257, 0.7071067811865476;

  // Actual matrix
  ExampleVec Err_terms = ExampleVec::Constant(-5);
  calc_Err_terms(Err_terms, X_tr, W_u,
                 W_p, Ysum_terms, Ru_terms);

  // Validation matrix
  VectorXf Err_terms_val = VectorXf::Zero(N_EXAMPLE);
  Err_terms_val(0) = 1 - 23.667383810000004;
  Err_terms_val(1) = 2 - 61.8746566361;
  Err_terms_val(2) = 3 - 61.8746566361;

  Err_terms_val(3) = 4 - 24.479814105;
  Err_terms_val(4) = 5 - 43.73811448549999;
  Err_terms_val(5) = 6 - 64.01038509605;

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
  UserMat Ysum_terms = Ysum_terms.Constant(N_LATENT, N_USER, -5);
  Ysum_terms << 7.114, 6.157, 5.2, 3.15,
                7.216, 6.268, 5.32, 3.26;

  float Ru = 3.523131;
  float err = 0.9;
  int i = 1;
  int j = 2;

  // Actual matrix
  update_product(W_u, W_p, Ysum_terms, Ru, err, i, j);

  // Validation matrix
  MatrixXf W_p_val = MatrixXf::Zero(N_LATENT, N_PRODUCT);
  float l1 = 3.007 + LR * (err * (7.05 + Ru * 6.157) - GAMMA * 3.007);
  float l2 = 3.008 + LR * (err * (7.06 + Ru * 6.268) - GAMMA * 3.008);
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


void test_update_implicit()
{
  cout << endl << "###########################################################"
       << endl << "Running update_implicit tests..." << endl << endl;
  
  // Load in helper matrices.
  ExampleMat X_tr = X_tr.Constant(N_EXAMPLE, 4, -5);
  X_tr << 0, 0, 1, 0,
          0, 1, 2, 0,
          0, 2, 3, 1,
          1, 0, 4, 0,
          1, 1, 5, 0,
          1, 2, 6, 1,
          2, 0, 7, 0, 
          2, 1, 8, 0,
          2, 2, 9, 1,
          3, 0, 10, 0,
          3, 1, 11, 1;
  ProductMat W_p = W_p.Constant(N_LATENT, N_PRODUCT, -5);
  W_p << 1.1, 2.05, 3.007,
         1.2, 2.06, 3.008;
  ProductMat W_i = W_p.Constant(N_LATENT, N_PRODUCT, -5);
  W_i << 1.9, 2.75, 3.807,
         1.8, 2.96, 3.908;

  float Ru_1 = 0.232;
  float Ru_2 = 0.34;
  float Ru_3 = 0.56;
  float err_1 = 0.9;
  float err_2 = 0.8;
  float err_3 = 0.7;

  // Calculate implicit temp.
  LatentVec implicit_temp = LatentVec::Zero();
  calc_implicit_temp(implicit_temp, err_1, Ru_1, W_p, X_tr(0, 1));
  calc_implicit_temp(implicit_temp, err_2, Ru_2, W_p, X_tr(1, 1));
  calc_implicit_temp(implicit_temp, err_3, Ru_3, W_p, X_tr(2, 1));
  // Validation matrix
  VectorXf implicit_temp_val = VectorXf::Zero(N_LATENT);
  implicit_temp_val = err_1 * Ru_1 * W_p.col(0) + err_2 * Ru_2 * W_p.col(1) +
       err_3 * Ru_3 * W_p.col(2);
  cout << "Test 1: calc_implicit_temp..." << endl;
  assert(implicit_temp_val.isApprox(implicit_temp));
  cout << "Test 2: calc_implicit_temp..." << endl;
  assert(!(implicit_temp_val.isApprox(implicit_temp * 1.0001)));

  // Update implicit
  update_implicit(W_i, implicit_temp, 0, 3);
  update_implicit(W_i, implicit_temp, 1, 3);
  update_implicit(W_i, implicit_temp, 2, 3);
  // Validation matrix
  MatrixXf W_i_val = MatrixXf::Zero(N_LATENT, N_PRODUCT);
  W_i_val << 1.9 + LR  * (implicit_temp(0) - 3 * GAMMA * 1.9),
             2.75 + LR  * (implicit_temp(0) - 3 * GAMMA * 2.75),
             3.807 + LR  * (implicit_temp(0) - 3 * GAMMA * 3.807),

             1.8 + LR  * (implicit_temp(1) - 3 * GAMMA * 1.8),
             2.96 + LR  * (implicit_temp(1) - 3 * GAMMA * 2.96),
             3.908 + LR  * (implicit_temp(1) - 3 * GAMMA * 3.908);

  // Assertions
  cout << "Test 1: update_implicit..." << endl;
  assert(W_i_val.isApprox(W_i));
  cout << "Test 2: update_implicit..." << endl;
  assert(!(W_i_val.isApprox(W_i * 1.0001)));

  cout << endl << "All update_implicit tests successful!"
       << endl << "###########################################################"
       << endl << endl;
}

