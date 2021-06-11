# Eigen3 SVD
Optimized-for-speed Eigen implementations of SVD, SVD++ and TimeSVD++ algorithms.

This repository is a redistribution of [Eric Zhao's SVDistic](https://github.com/ericzhao28/SVDistic).  
Original author is Eric Zhao, and this version differs from their only in build and Docker configurations.

![license](https://img.shields.io/github/license/mashape/apistatus.svg)
 
## Requirements
This application is fully Dockerized for easier usage.

If you wish to do not wish to deploy through Docker, you must have Eigen installed.
Find installation instructions here:
<http://eigen.tuxfamily.org/index.php?title=Main_Page>.

## Speed.
SVDistic was benchmarked on task of training an SVD and SVD++ model across 95 million examples (from the Netflix dataset) with 500,000 users and 18,000 products using 100 latent factors.
Each model was allocated a single 2.3 GHz Intel Xeon® E5-2686 v4 (Broadwell) core and each occupied a maximum of 1.7 GB of memory.
The SVD++ model completed a full epoch in 32.3 seconds.
The SVD model completed a full epoch in 11.3 seconds.

## Usage
Usage instructions for the actual Svdistic program.
```
Usage: ./svdistic <svd/svdpp/help> <train/infer/score>
Options: required settings are flaired with [r]
-model_id    STRING:  name of the model
-fname       STRING:  file name of data in data/corpus/
-n_product   INT:     number of products
-n_user      INT:     number of users
-n_example   INT:     number of examples to process
-report_freq INT:     frequency of epoch reports
-n_epochs    INT:     number of epochs to run training for
-reg_weight  FLOAT:   weight regularization strength
-reg_bias    FLOAT:   bias regularization strength
-lr          FLOAT:   learning rate
-lr_decay    FLOAT:   learning rate decay
```

Add your data files to /data/corpus and note the filename as command line arguments to the program. For training and validation, your data files must match the data format specified in the following section with three valid columns denoting user id, product id and ranking. For inference, your data file must still meet the data format, but fill in whatever you want for ranking.

## Docker
There are two Docker configurations for running and developing SVDistic respectively.  

### Running & Evaluation
```docker/prod/Dockerfile``` provides configuration for production image, which contains dependencies
required for building, deploying, and running SVDistic.

Create an image from this file and run a container with mounted dir (eg. ```your-local-project-dir:/svdistic```).
Then you can use SVDistic like it described in the Usage section.

### Development
```docker/dev/Dockerfile``` provides configuration for development image, which contains additional
dependencies and commands to run a Linux container for remote development on any machine.

Create an image from this file and run a container with forwarded 22 port, 
```SYS_PTRACE``` capability to enable debug, and some mounts if needed.
Example:
```shell
docker run
-p 127.0.0.1:2222:22
--name svdistic-dev
-d
--cap-add SYS_PTRACE
your-image-tag:latest 
```

Then you can develop SVDistic like on a remote server, using e.g. CLion configuration.


## Data format.
### Input data.
The standard data format is a csv file, where each entry corresponding to a ranking.
Do not add a header line.

Each line consists of 3 pairs: `user_index,product_index,score`.
To calculate the user index, transform user ids to indices ranging from 0 to the number of users.
The same process applies to product indexing.
Scores are expected to be integer values, although we weakly support float rankings.

Please note that **entries must be ordered by user index**. Entries with the same user index should be consecutive in the data file.

### Weight dumps.
Weights dumps are a column-major iteration through matrix
values. Every entry is separated by a newline.

