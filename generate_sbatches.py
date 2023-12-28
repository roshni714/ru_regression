import itertools
import glob
import os


SBATCH_PREFACE = """#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""

# constants for commands

OUTPUT_PATH = "/home/users/rsahoo/ru_regression/scripts/"
SAVE_PATH = "/home/users/rsahoo/ru_regression/"


def generate_runs():
    seed = 0
    #    gammas = [1., 1.5, 2.0, 2.5, 3.0]
    gammas = [1.0, 1.5, 2.0, 2.5, 3.0]
    losses = ["poisson_nll"]
    methods = ["joint_ru_regression", "ru_regression"]
    #    methods = ["ru_regression"]
    #    outcomes = ["marital", "vax_attitude_def", "any_ins"]
    #    gammas = [1., 1.2, 1.4]
    #    methods = ["lower_bound", "upper_bound", "ru_regression"]

    for loss in losses:
        for gamma in gammas:
            for method in methods:
                exp_id = "survey_{}_{}_{}_12-26-23".format(method, loss, gamma)
                script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                        ),
                        file=f,
                    )

                    base_cmd = "python train.py main --dataset survey --method {} --epochs 15 --seed {} --save survey --gamma {}  --loss {} --model_class neural_network --use_train_weights".format(
                        method, seed, gamma, loss
                    )
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)


generate_runs()
