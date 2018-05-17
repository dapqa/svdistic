#include "../types.h"

// Train function
void train(ExampleMat& X_tr, UserMat& W_u, ProductMat& W_i, ProductMat& W_p,
           UserVec& Ru_terms, UserMat& Ysum_terms);

// Weight init
void init_weights(UserMat& W_u, ProductMat& W_i, ProductMat& W_p);

// Term calculation
void calc_Ru_terms(UserVec& Ru_terms, ExampleMat& X_tr);
void calc_Ysum_terms(UserMat& Ysum_terms, ProductMat& W_i, ExampleMat& X_tr);
void calc_Pred_terms(ExampleVec& Pred_terms, ExampleMat& X_tr, UserMat& W_u,
                    ProductMat& W_p, UserMat& Ysum_terms, UserVec& Ru_terms);
void calc_Err_terms(ExampleVec& Err_terms, ExampleMat& X_tr, UserMat& W_u,
                    ProductMat& W_p, UserMat& Ysum_terms, UserVec& Ru_terms);

// Update rules
void update_user(UserMat& W_u, ProductMat& W_p, float err, int i, int j);
void update_product(UserMat& W_u, ProductMat& W_p, UserMat& Ysum,
                    float Ru, float err, int i, int j);

// Calculate implicit terms.
void calc_implicit_temp(LatentVec& implicit_temp, float err, float Ru,
                        ProductMat& W_p, int j);
void update_implicit(ProductMat& W_i, LatentVec& implicit_temp,
                     int j, float scale);

