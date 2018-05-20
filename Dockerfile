FROM ubuntu:16.04
MAINTAINER "Eric Zhao"

RUN apt-get update
RUN apt-get install -y wget tar libc6-dev build-essential cmake gcc g++ binutils make libeigen3-dev

WORKDIR /service
COPY ./ /service

CMD tail -f /dev/null
CMD make clean
CMD make

