#!/bin/bash

seeds=(0)
gammas=(2 4 8 16)
for SEED in "${seeds[@]}";
do
    python inference.py main --dataset shifted --method erm --loss squared_loss --seed $SEED --save sim
    python inference.py main --dataset shifted_oracle --method erm  --loss squared_loss --seed $SEED  --save sim
    for GAMMA in "${gammas[@]}";
    do
        python inference.py main --dataset shifted --method ru_regression --loss squared_loss --gamma $GAMMA --seed $SEED --save sim
    done
done

