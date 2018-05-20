#include "../types.h"

#ifndef _PIPES_H
#define _PIPES_H

// Options defined in pipes.cpp.
int input_pipeline(ExampleMat& X, int N_EXAMPLE, string f_name);
float load_float(string f_name);
void save_float(float v, string f_name);

// Load a matrix from a file.
// TODO: check if file exists and is legal.
template<class T, int rows, int cols>
void load_matrix(Matrix<T, rows, cols>& m, string f_name, int x, int y)
{
  float v;
  m.resize(x, y);
  int i = 0;
  FILE *f = fopen(f_name.c_str(), "r");
  while(fscanf(f, "%f", &v) != EOF)
  {
		m(i) = v;
    ++i;
	}
}

// Save a matrix into a file.
// TODO: check if directory exists.
template<class T, int rows, int cols>
void save_matrix(Matrix<T, rows, cols>& m, string f_name)
{
	IOFormat NewlineFmt(StreamPrecision, DontAlignCols,
											"\n", "\n", "", "", "", "");
	ofstream file(f_name.c_str());
  if (file.is_open())
  {
    file << m.transpose().format(NewlineFmt);
  }
}

#endif
