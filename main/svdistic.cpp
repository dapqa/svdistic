#include "../types.h"
#include "../models/base/base.h"
#include "../models/svd/model.h"
#include "../models/svdpp/model.h"
#include "./modes.cpp"


// Usage message in case of invalid arguments or command.
int usage_message()
{
  cout << "Usage: ./svdistic <svd/svdpp/help> <train/infer/score>" << endl
       << "Options: required settings are flaired with [r]" << endl
       << "-model_id    STRING:  name of the model"  << endl
       << "-fname       STRING:  file name of data in data/corpus/"  << endl
       << "-n_product   INT:     number of products"  << endl
       << "-n_user      INT:     number of users"  << endl
       << "-n_example   INT:     number of examples to process"  << endl
       << "-report_freq INT:     frequency of epoch reports"  << endl
       << "-n_epochs    INT:     number of epochs to run training for"  << endl
       << "-reg_weight  FLOAT:   weight regularization strength"  << endl
       << "-reg_bias    FLOAT:   bias regularization strength"  << endl
       << "-lr          FLOAT:   learning rate"  << endl
       << "-lr_decay    FLOAT:   learning rate decay"  << endl;
  return 1;
}


// Set model hyperparameters based on command line arguments.
int set_params(SVD& model, int argc, char* argv[], string& fname)
{ 
  // arg[0] = program, arg[1] = options, arg[2] = model.
  if (argc % 2 == 0)
    return 1;

  for (int i = 3; i < argc; i += 2)
  {
    if (strcmp(argv[i], "-model_id") == 0)
    {
      model.model_id = string (argv[i + 1]);
    }
    else if (strcmp(argv[i], "-fname") == 0)
    {
      fname = string (argv[i + 1]);
    }
    else if (strcmp(argv[i], "-n_product") == 0)
    {
      istringstream in(argv[i + 1]);
      in >> model.N_PRODUCT;
    }
    else if (strcmp(argv[i], "-n_user") == 0)
    {
      istringstream in(argv[i + 1]);
      in >> model.N_USER;
    }
    else if (strcmp(argv[i], "-n_example") == 0)
    {
      istringstream in(argv[i + 1]);
      in >> model.N_EXAMPLE;
    }
    else if (strcmp(argv[i], "-report_freq") == 0)
    {
      istringstream in(argv[i + 1]);
      in >> model.REPORT_FREQ;
    }
    else if (strcmp(argv[i], "-n_epochs") == 0)
    {
      istringstream in(argv[i + 1]);
      in >> model.N_EPOCHS;
    }
    else if (strcmp(argv[i], "-reg_bias") == 0)
    {
      istringstream in(argv[i + 1]);
      in >> model.REG_B;
    }
    else if (strcmp(argv[i], "-reg_weight") == 0)
    {
      istringstream in(argv[i + 1]);
      in >> model.REG_W;
    }
    else if (strcmp(argv[i], "-lr") == 0)
    {
      istringstream in(argv[i + 1]);
      in >> model.LR;
    }
    else if (strcmp(argv[i], "-lr_decay") == 0)
    {
      istringstream in(argv[i + 1]);
      in >> model.LR_DECAY;
    }
    else
    {
      return 1;
    }
  }
  return 0;
}


template <class T>
int mode_handler(char* argv[], T& model, string fname)
{
  // Depending on the avaiable modes, act on the
  // selected model.
  if (!strcmp(argv[2], "train"))
    return train<T>(model, fname);
  if (!strcmp(argv[2], "infer"))
    return infer<T>(model, fname);
  if (!strcmp(argv[2], "score"))
    return score<T>(model, fname);

  // Invalid mode option.
  return usage_message();
}


// Main function.
int main(int argc, char* argv[])
{
  // File name.
  string fname = "dummy.data";

  // Ensure program is available.
  if (argc == 1) return usage_message();

  // Various program options
  if (!strcmp(argv[1], "help"))
  {
    // If "help".
    usage_message();
    return 0;
  }
  else if (!strcmp(argv[1], "svd"))
  {
    // If they want a SVD.
    SVD model;
    model.model_id = "default_svd";
    model.N_PRODUCT = 3;
    model.N_USER = 4;
    model.N_EXAMPLE = 11;
    model.REPORT_FREQ = 2;
    model.N_EPOCHS = 10;
    model.REG_B = 0.005;
    model.REG_W = 0.015;
    model.LR = 0.001;
    model.LR_DECAY = 0.98;

    // If not help message, require model mode.
    if (argc == 2) return usage_message();

    // Load in hyperparameters.
    if (set_params(model, argc, argv, fname) == 1)
      return usage_message();

    return mode_handler<SVD>(argv, model, fname);
  }
  else if (!strcmp(argv[1], "svdpp"))
  {
    // If they want a SVD++.
    SVDpp model;
    model.model_id = "default_svdpp";
    model.N_PRODUCT = 3;
    model.N_USER = 4;
    model.N_EXAMPLE = 11;
    model.REPORT_FREQ = 2;
    model.N_EPOCHS = 10;
    model.REG_B = 0.005;
    model.REG_W = 0.015;
    model.LR = 0.001;
    model.LR_DECAY = 0.98;

    // If not help message, require model mode.
    if (argc == 2) return usage_message();

    // Load in hyperparameters.
    if (set_params(model, argc, argv, fname) == 1)
      return usage_message();

    return mode_handler<SVDpp>(argv, model, fname);
  }
  else
  {
    // If invalid program options.
    return usage_message();
  }
}

