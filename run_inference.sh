#!/bin/bash
MODEL_PATH=`/scratch/users/rsahoo/models"
seeds=(0)
gammas=(2 4 8 16)
for SEED in "${seeds[@]}";
do
    python inference.py main --dataset shifted_one_dim --method erm  --loss squared_loss --seed $SEED  --save inference --p_train 0.5 --model_path $MODEL_PATH
    python inference.py main --dataset shifted_one_dim --method erm --loss squared_loss --seed $SEED --save inference --p_train 0.2 --model_path $MODEL_PATH
    for GAMMA in "${gammas[@]}";
    do
        python inference.py main --dataset shifted_one_dim --method ru_regression --loss squared_loss --gamma $GAMMA --seed $SEED --save inference --p_train 0.2 --model_path $MODEL_PATH
    done
done

