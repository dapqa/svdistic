#include "./pipes.h"


int input_pipeline(ExampleMat& X_tr, string trainFile)
{
  // Load file
  FILE *fp = fopen(trainFile.c_str(), "r");
  // Staging objs we'll encounter
  int i, j, y, ij;
  int past_i = -1;

  // Iterate through training data
  X_tr = X_tr.Constant(N_EXAMPLE, 4, EMPTY_VAL);
  ij = 0;
  while(fscanf(fp, "%d,%d,%d", &i, &j, &y) != EOF)
  {
    // First index is user_id
    X_tr(ij, 0) = i;
    // Second index is product_id
    X_tr(ij, 1) = j;
    // Third index is the score
    X_tr(ij, 2) = y;

    // If the past_i has been updated
    // and past_i is different from this i
    if ((past_i != -1) && (past_i != i))
    {
      // Then we've changed users.
      // We note the last index's 4th index
      // is 1.
      X_tr(ij - 1, 3) = 1;
    }

    past_i = i;
    ++ij;
  }

  X_tr(ij - 1, 3) = 1;

  // Make sure we were given the right number of examples.
  if (ij != N_EXAMPLE)
  {
    return 1;
  }

  // Close file
  fclose(fp);
  return 0;
}

void load_weights(UserMat& W_u, ProductMat& W_i,
                   ProductMat& W_p, UserVec& Ru_terms,
                   UserMat& Ysum_terms, string W_u_f,
                   string W_i_f, string W_p_f,
                   string Ru_f, string Ysum_f)
{
  int i;
  float v;

  W_i = ProductMat::Zero(N_LATENT, N_PRODUCT);

  // Load W_i
  FILE *fp = fopen(W_i_f.c_str(), "r");
	i = 0;
  while(fscanf(fp, "%f", &v) != EOF)
  {
		W_i(i) = v;
		++i;
	}

  Ru_terms = UserVec::Zero();
  fp = fopen(Ru_f.c_str(), "r");
	i = 0;
  while(fscanf(fp, "%f", &v) != EOF)
  {
		Ru_terms(i) = v;
		++i;
	}

  Ysum_terms = UserMat::Zero(N_LATENT, N_USER);
  fp = fopen(Ysum_f.c_str(), "r");
	i = 0;
  while(fscanf(fp, "%f", &v) != EOF)
  {
		Ysum_terms(i) = v;
		++i;
	}

	load_weights(W_u, W_p, W_u_f, W_p_f);
}

void load_weights(UserMat& W_u, ProductMat& W_p,
                  string W_u_f, string W_p_f)
{
  int i;
  float v;

  W_u = UserMat::Zero(N_LATENT, N_USER);
  W_p = ProductMat::Zero(N_LATENT, N_PRODUCT);

  // Load W_u
  FILE *fp = fopen(W_u_f.c_str(), "r");
	i = 0;
  while(fscanf(fp, "%f", &v) != EOF)
  {
		W_u(i) = v;
		++i;
	}

  // Load W_p
  fp = fopen(W_p_f.c_str(), "r");
	i = 0;
  while(fscanf(fp, "%f", &v) != EOF)
  {
		W_p(i) = v;
		++i;
	}
}


void save_pipeline(UserMat& W_u, ProductMat& W_i,
                   ProductMat& W_p, UserVec& Ru_terms,
                   UserMat& Ysum_terms, string W_u_f,
                   string W_i_f, string W_p_f,
                   string Ru_f, string Ysum_f)
{
	IOFormat NewlineFmt(StreamPrecision, DontAlignCols,
												"\n", "\n", "", "", "", "");
	ofstream file(W_i_f.c_str());
  if (file.is_open())
  {
    file << W_i.transpose().format(NewlineFmt);
  }
	ofstream file1(Ru_f.c_str());
  if (file1.is_open())
  {
    file1 << Ru_terms.format(NewlineFmt);
  }
	ofstream file2(Ysum_f.c_str());
  if (file2.is_open())
  {
    file2 << Ysum_terms.transpose().format(NewlineFmt);
  }
  save_pipeline(W_u, W_p, W_u_f, W_p_f);
}


void save_pipeline(UserMat& W_u, ProductMat& W_p,
                   string W_u_f, string W_p_f)
{
	IOFormat NewlineFmt(StreamPrecision, DontAlignCols,
												"\n", "\n", "", "", "", "");
	ofstream file(W_u_f.c_str());
  if (file.is_open())
  {
    file << W_u.transpose().format(NewlineFmt);
  }
	ofstream file2(W_p_f.c_str());
  if (file2.is_open())
  {
    file2 << W_p.transpose().format(NewlineFmt);
  }
}

void save_pipeline(ExampleVec& Pred_terms, string p_f)
{
	IOFormat NewlineFmt(StreamPrecision, DontAlignCols,
												"\n", "\n", "", "", "", "");
	ofstream file(p_f.c_str());
  if (file.is_open())
  {
    file << Pred_terms.format(NewlineFmt);
  }
}

