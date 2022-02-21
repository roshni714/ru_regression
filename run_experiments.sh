#!/bin/bash

python train.py main --dataset simulated --method erm --epochs 200 --loss squared_loss --seed 0
python train.py main --dataset simulated --method ru_regression --epochs 200 --loss squared_loss --gamma 2 --seed 0

