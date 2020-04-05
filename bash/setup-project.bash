#!/bin/bash

# Create build folder
cd ../
mkdir cmake-build-debug
cd cmake-build-debug/
cmake ../ -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles"

# Compile dml library
cd ../
# If you have the resources and you want to compile faster the library, please uncomment and comment the commands respectively
#cmake --build cmake-build-debug --target dml -- -j 6
cmake --build cmake-build-debug --target dml
