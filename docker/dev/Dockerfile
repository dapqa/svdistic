FROM ubuntu:16.04
LABEL org.opencontainers.image.authors="Alexander Nechaev <dapqa@yandex.ru>"

RUN apt-get update && apt-get install -y wget tar libc6-dev build-essential cmake gcc g++ gdb binutils make libeigen3-dev python3
RUN apt-get install -y ssh
RUN useradd -m user && yes password | passwd user

ENTRYPOINT service ssh start && /bin/bash