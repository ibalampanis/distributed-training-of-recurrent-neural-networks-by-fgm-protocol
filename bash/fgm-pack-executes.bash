#!/bin/bash

mkdir out 

for ((i=1; i<=10; i++)); do
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_1.json >> out/fgm_exp1_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_2.json >> out/fgm_exp2_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_3.json >> out/fgm_exp3_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_4.json >> out/fgm_exp4_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_5.json >> out/fgm_exp5_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_6.json >> out/fgm_exp6_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_7.json >> out/fgm_exp7_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_9.json >> out/fgm_exp9_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_10.json >> out/fgm_exp10_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_11.json >> out/fgm_exp11_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_12.json >> out/fgm_exp12_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_13.json >> out/fgm_exp13_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_15.json >> out/fgm_exp15_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_16.json >> out/fgm_exp16_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_17.json >> out/fgm_exp17_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_18.json >> out/fgm_exp18_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_20.json >> out/fgm_exp20_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_21.json >> out/fgm_exp21_iter${i}.txt &
    ./FGMLearn /tmp/simulation/fgm-set-1/fgm_set1_23.json >> out/fgm_exp23_iter${i}.txt &
    wait
done
