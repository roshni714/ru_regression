#!/bin/bash

seeds=(0)
gammas=(2 4 8 16)
for SEED in "${seeds[@]}";
do
    python inference.py main --dataset shifted_one_dim --method erm --loss squared_loss --seed $SEED --save inference_res
    python inference.py main --dataset shifted_one_dim --method erm  --loss squared_loss --seed $SEED  --save inference_res
    for GAMMA in "${gammas[@]}";
    do
        python inference.py main --dataset shifted_one_dim --method ru_regression --loss squared_loss --gamma $GAMMA --seed $SEED --save inference_res
    done
done

