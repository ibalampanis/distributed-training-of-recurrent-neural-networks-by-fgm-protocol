#!/bin/bash

mkdir out

cd ../

for ((i=1; i<=10; i++)); do
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_1.json >> bash/out/exp1_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_2.json >> bash/out/exp2_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_3.json >> bash/out/exp3_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_4.json >> bash/out/exp4_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_5.json >> bash/out/exp5_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_6.json >> bash/out/exp6_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_7.json >> bash/out/exp7_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_9.json >> bash/out/exp9_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_10.json >> bash/out/exp10_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_11.json >> bash/out/exp11_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_12.json >> bash/out/exp12_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_13.json >> bash/out/exp13_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_15.json >> bash/out/exp15_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_16.json >> bash/out/exp16_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_17.json >> bash/out/exp17_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_18.json >> bash/out/exp18_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_20.json >> bash/out/exp20_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_21.json >> bash/out/exp21_iter${i}.txt
    ./cmake-build-debug/bin/GMLearn /home/ibalampanis/tests/simulation/gm-set-1/gm_set1_23.json >> bash/out/exp23_iter${i}.txt
done




