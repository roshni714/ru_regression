#!/bin/bash

P_TRAIN=0.2
P_TEST_LO=0.1
P_TEST_HI=0.8
p_tests=(0.1 0.275 0.45 0.625 0.8)

seeds=(4 5 6 7 8 9)
gammas=(2 4 8 16)

for SEED in "${seeds[@]}";
do
    for P_TEST in "${p_tests[@]}";
    do
        python train.py main --dataset shifted_one_dim --p_test_lo $P_TEST --p_test_hi $P_TEST --p_train $P_TEST --n_test_sweep 1 --method erm --epochs 100 --loss squared_loss --seed $SEED --save sim_sweep_p_test
    done
    python train.py main --dataset shifted_one_dim --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI --p_train $P_TRAIN --n_test_sweep 5 --method erm --epochs 100 --loss squared_loss --seed $SEED --save sim_sweep_p_test
    for GAMMA in "${gammas[@]}";
    do
        python train.py main --dataset shifted_one_dim --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI --p_train $P_TRAIN --n_test_sweep 5 --method ru_regression --epochs 100 --loss squared_loss --gamma $GAMMA --seed $SEED --save sim_sweep_p_test
    done

done


