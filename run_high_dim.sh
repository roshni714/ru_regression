#!/bin/bash
RUN_PATH='/scratch/users/rsahoo/runs'
MODEL_PATH='/scratch/users/rsahoo/models'
P_TRAIN=0.2
P_TEST_LO=0.1
P_TEST_HI=0.9
p_tests=(0.1 0.5 0.7 0.9)
D=16
seeds=(0 1 2 3 4 5)
gammas=(2 4 8 16)

for SEED in "${seeds[@]}";
do
    # Oracle ERM
    for P_TEST in "${p_tests[@]}";
    do
        python train.py main --n_train 100000 --dataset shifted_high_dim --d $D --p_test_lo $P_TEST --p_test_hi $P_TEST --p_train $P_TEST --n_test_sweep 1 --method erm --epochs 50 --loss squared_loss --seed $SEED --save sim_high_dim_2 --run_path $RUN_PATH --model_path $MODEL_PATH

    done
     
    # Standard ERM
    python train.py main --n_train 100000 --dataset shifted_high_dim --d $D --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI --p_train $P_TRAIN --n_test_sweep 5 --method erm --epochs 50 --loss squared_loss --seed $SEED --save sim_high_dim_2 --run_path $RUN_PATH --model_path $MODEL_PATH


    # RU Regression
    for GAMMA in "${gammas[@]}";
    do
        python train.py main --n_train 100000 --dataset shifted_high_dim --d $D --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI --p_train $P_TRAIN --n_test_sweep 5 --method ru_regression --epochs 50 --loss squared_loss --gamma $GAMMA --seed $SEED --save sim_high_dim_2 --run_path $RUN_PATH --model_path $MODEL_PATH

    done

done


