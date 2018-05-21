#include "../../types.h" 

#ifndef _BASE_H
#define _BASE_H


class Base
{
  public:
    // Public functions for the three standard operations.
    void train(ExampleMat& X_tr);
    void infer(ExampleVec& preds, ExampleMat& X_test);
    float score(ExampleMat& X_val);

    // Allow public to manipulate weights.
    virtual void init_weights() = 0;
    virtual void load_weights() = 0;
    virtual void save_weights() = 0;

    // Model_id to be define upon instantation.
    // Used in model saving and dumping.
    string model_id;
    int N_EXAMPLE;

    // Settings
    int N_PRODUCT;
    int N_USER;
    int REPORT_FREQ;

    // Hyperparameters
    int N_EPOCHS;
    float REG_B;
    float REG_W;
    float LR;

  protected:
    // pred_diff = pred - truth.
    float pred_diff(ExampleMat& X, int ij);
    // Calculate global mean.
    void calc_mu(ExampleMat& X);
    // Calculate user mean.
    void user_bias(float err, ExampleMat& X, int ij);
    // Calculate product mean.
    void product_bias(float err, ExampleMat& X, int ij);

    // Predict value for a given term.
    virtual float predict(ExampleMat& X, int ij) = 0;
    // Update weights for a given term.
    virtual void update(ExampleMat& X, int ij) = 0;

    // Bias weights.
    UserVec b_u;
    ProductVec b_p;
    float mu;
};

#endif
