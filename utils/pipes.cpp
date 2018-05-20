#include "./pipes.h"


// Load file into ExampleMat.
int input_pipeline(ExampleMat& X, int N_EXAMPLE, string f_name)
{
  // Load file
  FILE *fp = fopen(f_name.c_str(), "r");
  if (fp == NULL)
  {
    cout << "Data file does not exist." << endl;
    return 1;
  }

  // Staging objs we'll encounter
  int i, j, y;
  int ij = 0;

  // Scan in data. We assume X is initialized.
  while(fscanf(fp, "%d,%d,%d", &i, &j, &y) != EOF)
  {
    // First index is user_id
    X(0, ij) = i;
    // Second index is product_id
    X(1, ij) = j;
    // Third index is the score
    X(2, ij) = y;
    ++ij;
  }

  // Make sure we were given the right number of examples.
  if (ij != N_EXAMPLE)
  {
    cout << "Invalid number of testing examples." << endl;
    return 1;
  }

  // Close file
  fclose(fp);
  return 0;
}


// Save floats into file.
// TODO: check path existence.
void save_float(float v, string f_name)
{
  FILE *f = fopen(f_name.c_str(), "w");
  fprintf(f, "%f", v);
  fclose(f);
}


// Load floats from file.
// TODO: check file existence and legality.
float load_float(string f_name)
{
  float v;
  FILE *f = fopen(f_name.c_str(), "r");
  assert(fscanf(f, "%f", &v) != EOF);
  return v;
}

