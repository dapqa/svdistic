FROM ubuntu:16.04
MAINTAINER "Eric Zhao"

RUN apt-get update && apt-get install -y wget tar libc6-dev build-essential cmake gcc g++ binutils make libeigen3-dev python3

WORKDIR /service

RUN wget https://s3-us-west-2.amazonaws.com/cs156preprocessed/corpus.tgz
RUN tar -zxvf ./corpus.tgz
RUN mkdir /service/data
RUN mv /service/corpus /service/data/

COPY ./ /service

ARG n_latent
ENV n_latent=$n_latent
CMD ["/bin/bash", "-c", "echo \"static const int N_LATENT = $n_latent;\" > ./config.h"]
RUN make clean; exit 0
RUN make

ARG run_cmd
ENV run_cmd=$run_cmd
CMD ["/bin/bash", "-c", "$run_cmd"]

