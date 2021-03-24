cmake_minimum_required(VERSION 3.5)
project(SVDISTIC)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories("/usr/include/eigen3/")

add_executable(svdistic
        main/svdistic.cpp
        models/svdpp/model.cpp
        models/svd/model.cpp
        models/base/base.cpp
        utils/pipes.cpp
        )