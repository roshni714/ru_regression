#!/bin/bash

rm -rf scripts
mkdir scripts
python generate_sbatches.py
rm -rf /scratch/users/rsahoo/models
rm -rf /scratch/users/rsahoo/runs
rm -rf results
mkdir results


for experiment in /home/users/rsahoo/ru_regression/scripts/*.sh
do
    echo $experiment
    chmod u+x $experiment
    sbatch $experiment
#    $experiment
    sleep 1
done

echo "Done"
