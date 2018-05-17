#include "../types.h"

// Train function
void train(ExampleMat& X_tr, UserMat& W_u, ProductMat& W_p);

// Weight init
void init_weights(UserMat& W_u, ProductMat& W_p);

// Term calculation
void calc_Pred_terms(ExampleVec& Pred_terms, ExampleMat& X_tr, UserMat& W_u,
                     ProductMat& W_p);
void calc_Err_terms(ExampleVec& Err_terms, ExampleMat& X_tr, UserMat& W_u,
                    ProductMat& W_p);

// Update rules
void update_user(UserMat& W_u, ProductMat& W_p, float err, int i, int j);
void update_product(UserMat& W_u, ProductMat& W_p, float err, int i, int j);

