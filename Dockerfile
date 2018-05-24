FROM ubuntu:16.04
MAINTAINER "Eric Zhao"

RUN apt-get update
RUN apt-get install -y wget tar libc6-dev build-essential cmake gcc g++ binutils make libeigen3-dev python3

WORKDIR /service

RUN wget https://s3-us-west-2.amazonaws.com/cs156preprocessed/corpus.tgz
RUN tar -zxvf ./corpus.tgz

COPY ./ /service

RUN mv /service/corpus/* /service/data/corpus/

# RUN mv ./mu/all.dta /service/preprocessing/
# RUN mv ./mu/all.idx /service/preprocessing/
# RUN rm -rf ./mu/

# WORKDIR /service/preprocessing
# RUN bash process.sh
# RUN mv /service/preprocessing/base.data /service/data/corpus/base.data
# RUN mv /service/preprocessing/probe.data /service/data/corpus/probe.data
# RUN mv /service/preprocessing/qual.data /service/data/corpus/qual.data

RUN make clean; exit 0
RUN make

WORKDIR /service
CMD bash run.sh

