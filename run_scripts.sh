#!/bin/bash

rm -rf scripts
mkdir scripts
python generate_sbatches.py
rm -rf results
mkdir results


for experiment in /zfs/gsb/intermediate-yens/rsahoo/ru_regression/scripts/*.sh
do
    echo $experiment
    chmod u+x $experiment
    sbatch $experiment
    #$experiment
    sleep 1
done

echo "Done"
