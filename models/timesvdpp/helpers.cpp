/***********************************************
 * Higher level training steps
 **********************************************/

// Initialize on corpus
void TimeSVDpp::per_corpus(ExampleMat& X)
{
  SVDpp:per_corpus(X);
}

// Initialize for an epoch
void TimeSVDpp::per_epoch(ExampleMat& X)
{
  SVDpp:per_epoch(X);
}

// Initalize user info
void TimeSVDpp::init_user(ExampleMat& X, int ij)
{
  SVDpp:per_user(X, ij);
}

// End user
void TimeSVDpp::end_user(ExampleMat& X, int ij)
{
  SVDpp:per_user(X, ij);
}

// Initialize for an element.
void TimeSVDpp::update(ExampleMat& X, int ij)
{
  SVDpp:per_user(X, ij);
}


/***********************************************
 * Saving/loading...
 **********************************************/

// Initialize all weights.
void TimeSVDpp::init_weights()
{
  SVDpp::init_weights();
}

// Save weights into file.
void TimeSVDpp::save_weights()
{
  SVDpp::save_weights();
}

// Load weights from file.
void TimeSVDpp::load_weights()
{
  SVDpp::load_weights();
}

