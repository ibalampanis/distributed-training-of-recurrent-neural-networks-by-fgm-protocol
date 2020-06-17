#!/bin/bash

mkdir out 

for ((i=1; i<=10; i++)); do
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_1.json >> out/gm_exp1_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_2.json >> out/gm_exp2_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_3.json >> out/gm_exp3_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_4.json >> out/gm_exp4_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_5.json >> out/gm_exp5_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_6.json >> out/gm_exp6_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_7.json >> out/gm_exp7_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_9.json >> out/gm_exp9_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_10.json >> out/gm_exp10_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_11.json >> out/gm_exp11_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_12.json >> out/gm_exp12_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_13.json >> out/gm_exp13_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_15.json >> out/gm_exp15_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_16.json >> out/gm_exp16_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_17.json >> out/gm_exp17_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_18.json >> out/gm_exp18_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_20.json >> out/gm_exp20_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_21.json >> out/gm_exp21_iter${i}.txt &
    ./GMLearn /tmp/simulation/gm-set-1/gm_set1_23.json >> out/gm_exp23_iter${i}.txt &
    wait
done