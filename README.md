# SVD, SVD++ implementations.
Implementation of rank-approximation SVD and SVD++ algorithms on Eigen.

![license](https://img.shields.io/github/license/mashape/apistatus.svg)
[![Maintenance Intended](http://maintained.tech/badge.svg)](http://maintained.tech/)

## Requirements
You must have Eigen installed. Find installation instructions here:
<http://eigen.tuxfamily.org/index.php?title=Main_Page>.
We intend to Dockerize this package for easier distribution in the near future.

## Data format.
The standard data format is: `user_index,product_index,int_score`.
Entries with the same `user_index` should be consecutive.
Ensure `user_index` is within 0 and `N_USER`,
`product_index` within 0 and `N_PRODUCT`, and y is within 1 and 5.
Entires of a single user must be consecutive.

Weights dumps are a column-major iteration through matrix
values, where each line corresponds to the value.

## Usage.
`make all` first.

For training, add your data file to `data/train.txt`,
update `types.h` to match your `N_USER`, `N_EXAMPLE`, `N_PRODUCT`
and specify hyperparameters. Then, run `./svd -train` or
`./svdpp -train`.

For inference, add your data file to `data/to_infer.txt`,
fill in whatever values you want in the 3rd index of each
line (where score would usually be).
Update `types.h` to match your `N_USER`, `N_EXAMPLE`, `N_PRODUCT`
and specify hyperparameters. Then, run `./svd -infer` or
`./svdpp -infer`. The inference file is in "data/inferred.txt".

For RMSE scoring, add your data file to `data/validation.txt`,
update `types.h` to match your `N_USER`, `N_EXAMPLE`, `N_PRODUCT`
and specify hyperparameters. Then, run `./svd -validation` or
`./svdpp -validation`. The RMSE will be streamed to stdout.

## Todo
* Complete TimeSVD++ implementation.
* Add proper error messages, rather than allowing program
  to naturally fail.
* Bundle into Python package & make hyperparameters togglable.
* Dockerize this C++ library for easier distribution.

## Tests
We currently offer 3 sets of unit tests: one for SVD
logic, SVD++ logic, and utilities logic. To run the
test suite, replace `types.h` with `test_types.h`.
Build with `make all` and run `./test_svd` and `./test_svdpp`.


