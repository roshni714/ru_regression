#!/bin/bash

python train.py main --dataset simulated --epochs 200 --loss squared_loss --gamma 2 --seed 0
