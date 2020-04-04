#!/bin/bash

# Create folders and the file of library to be found by cmake
cd ../
mkdir cmake-build-debug
mkdir cmake-build-debug/cpp
mkdir cmake-build-debug/cpp/experiments
cd cmake-build-debug/cpp/experiments
touch libdml.a

# Generate cmake files
cd ../../
cmake ../ -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles"

# Compile dml library
cd ../
# If you have the resources and you want to compile faster the library, please uncomment and comment the commands respectively
#cmake --build cmake-build-debug --target dml -- -j 6
cmake --build cmake-build-debug --target dml
