#!/bin/bash

seeds=(3)
gammas=(2 4 8 16)
for SEED in "${seeds[@]}";
do
    python train.py main --dataset shifted --method erm --epochs 100 --loss squared_loss --seed $SEED --save sim2
    python train.py main --dataset shifted_oracle --method erm --epochs 100 --loss squared_loss --seed $SEED  --save sim2
#    for GAMMA in "${gammas[@]}";
#    do
#        python train.py main --dataset shifted --method ru_regression --epochs 100 --loss squared_loss --gamma $GAMMA --seed $SEED --save sim2
#    done
done

