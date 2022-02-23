#!/bin/bash

seeds=(0 1 2 3 4 5 6 7 8 9)
gammas=(2 4 8 16)
for SEED in "${seeds[@]}";
do
    python train.py main --dataset shifted --method erm --epochs 100 --loss squared_loss --seed $SEED --save sim
    python train.py main --dataset shifted_oracle --method erm --epochs 100 --loss squared_loss --seed $SEED  --save sim
    python inference.py main --dataset shifted --method erm --loss squared_loss --seed $SEED --save sim
    python inference.py main --dataset shifted_oracle --method erm  --loss squared_loss --seed $SEED  --save sim
    for GAMMA in "${gammas[@]}";
    do
        python train.py main --dataset shifted --method ru_regression --epochs 100 --loss squared_loss --gamma $GAMMA --seed $SEED --save sim
        python inference.py main --dataset shifted --method ru_regression --loss squared_loss --gamma $GAMMA --seed $SEED --save sim

    done
done
