#!/bin/bash

# >>

mkdir out
rm -f out/*.txt

cd ../

./cmake-build-debug/bin/GMLearn /home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/input_files/sample.json >> bash/out/exp1.txt &
./cmake-build-debug/bin/GMLearn /home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/input_files/sample.json >> bash/out/exp2.txt &
./cmake-build-debug/bin/GMLearn /home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/input_files/sample.json >> bash/out/exp3.txt &