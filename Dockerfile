FROM ubuntu:16.04
MAINTAINER "Eric Zhao"

RUN apt-get update
RUN apt-get install -y wget tar libc6-dev build-essential cmake gcc g++ binutils make libeigen3-dev python3

WORKDIR /service

RUN wget https://s3-us-west-2.amazonaws.com/cs156preprocessed/corpus.tgz
RUN tar -zxvf ./corpus.tgz

COPY ./ /service

RUN mv /service/corpus/* /service/data/corpus/

RUN make clean; exit 0
RUN make

WORKDIR /service
RUN chmod +x ./run.sh
ARG index
ENV idx=$index
CMD ["/bin/bash", "-c", "./run.sh $idx"]

