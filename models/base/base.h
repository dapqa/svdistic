#include "../../types.h" 
#include "../../utils/pipes.h"


#ifndef _BASE_H
#define _BASE_H


class Base
{
  public:
    // Public functions for the three standard operations.
    void train(ExampleMat& X_tr);
    void infer(ExampleVec& preds, ExampleMat& X_test);
    float score(ExampleMat& X_val);

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
    float LR_DECAY;
    // Manipulate weights
    void init_weights();
    void load_weights();
    void save_weights();

  protected:
    // pred_diff = pred - truth.
    float pred_diff(ExampleMat& X, int ij);

    // Update weights
    void calc_mu(ExampleMat& X);
    void user_bias(float err, ExampleMat& X, int ij);
    void product_bias(float err, ExampleMat& X, int ij);

    // Predict and update.
    virtual float predict(ExampleMat& X, int ij) = 0;
    virtual void per_corpus(ExampleMat& X) = 0;
    virtual void per_epoch(ExampleMat& X) = 0;
    virtual void init_user(ExampleMat& X, int ij) = 0;
    virtual void end_user(ExampleMat& X, int ij) = 0;
    virtual void update(ExampleMat& X, int ij) = 0;

    // Model weights.
    UserVec b_u;
    ProductVec b_p;
    float mu;
};

#endif
