#ifndef _SVDPP_H
#define _SVDPP_H
#include "model.h"
#include "helpers.cpp"


// Initialize all weights.
void TimeSVDpp::init_weights()
{
  a_u = UserVec::Random();
  b_p_t = ProductVec::Random(); 
  b_u_t = UserVec::Random(); 
  SVDpp::init_weights();
}


// Save weights into file.
void TimeSVDpp::save_weights()
{
  save_matrix<float, N_PRODUCT, 1>(b_p_t, "weights/" + model_id + "-b_p_t", N_USER, 1);
  save_matrix<float, N_USER, 1>(b_u_t, "weights/" + model_id + "-b_u_t", N_USER, 1);
  save_matrix<float, N_USER, 1>(a_u, "weights/" + model_id + "-a_u", N_USER, 1);
  SVDpp::save_weights();
}


// Load weights from file.
void TimeSVDpp::load_weights()
{
  load_matrix<float, N_PRODUCT, 1>(b_p_t, "weights/" + model_id + "-b_p_t", N_USER, 1);
  load_matrix<float, N_USER, 1>(b_u_t, "weights/" + model_id + "-b_u_t", N_USER, 1);
  load_matrix<float, N_USER, 1>(a_u, "weights/" + model_id + "-a_u", N_USER, 1);
  SVDpp::load_weights();
}

