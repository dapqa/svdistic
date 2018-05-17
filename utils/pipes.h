#include "../types.h"


int input_pipeline(ExampleMat& X_tr, string trainFile);
void load_weights(UserMat& W_u, ProductMat& W_i,
                   ProductMat& W_p, UserVec& Ru_terms,
                   UserMat& Ysum_terms, string W_u_f,
                   string W_i_f, string W_p_f,
                   string Ru_f, string Ysum_f);
void load_weights(UserMat& W_u, ProductMat& W_p,
                  string W_u_f, string W_p_f);
void save_pipeline(UserMat& W_u, ProductMat& W_i,
                   ProductMat& W_p, UserVec& Ru_terms,
                   UserMat& Ysum_terms, string W_u_f,
                   string W_i_f, string W_p_f,
                   string Ru_f, string Ysum_f);
void save_pipeline(UserMat& W_u, ProductMat& W_p,
                   string W_u_f, string W_p_f);
void save_pipeline(ExampleVec& Pred_terms, string p_f);

