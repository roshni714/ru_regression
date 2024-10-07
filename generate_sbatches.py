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


def generate_survey_runs():
    seed = 0
    gammas = [1.0, 1.5, 2.0, 2.5, 3.0]
    losses = ["poisson_nll"]
    methods = ["joint_ru_regression", "ru_regression"]

    for loss in losses:
        for gamma in gammas:
            for method in methods:
                exp_id = "survey_linear_{}_{}_{}_12-26-23".format(method, loss, gamma)
                script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                        ),
                        file=f,
                    )

                    base_cmd = "python train.py main --dataset survey --method {} --epochs 20 --seed {} --save survey --gamma {}  --loss {} --model_class linear".format(
                        method, seed, gamma, loss
                    )
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)

                    if method == "ru_regression":
                        base_cmd = "python train.py main --dataset survey --method {} --epochs 20 --seed {} --save survey --gamma {}  --loss {} --model_class linear --use_train_weights".format(
                            method, seed, gamma, loss
                        )
                        print(base_cmd, file=f)
                        print("sleep 1", file=f)

def generate_survey_nn_runs():
    seed = 0
    gammas = [1.0, 1.5, 2.0, 2.5, 3.0]
    losses = ["poisson_nll"]
    methods = ["joint_ru_regression", "ru_regression"]

    for loss in losses:
        for gamma in gammas:
            for method in methods:
                exp_id = "survey_nn_{}_{}_{}_12-26-23".format(method, loss, gamma)
                script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                        ),
                        file=f,
                    )

                    base_cmd = "python train.py main --dataset survey --method {} --epochs 20 --seed {} --save survey --gamma {}  --loss {} --model_class neural_network".format(
                        method, seed, gamma, loss
                    )
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)

                    if method == "ru_regression":
                        base_cmd = "python train.py main --dataset survey --method {} --epochs 20 --seed {} --save survey --gamma {}  --loss {} --model_class neural_network --use_train_weights".format(
                            method, seed, gamma, loss
                        )
                        print(base_cmd, file=f)
                        print("sleep 1", file=f)



def generate_severe_mimic_runs():
    seed = 0
    gammas = [1.0, 2.0, 4.0, 8.0, 16.0]
    losses = ["squared_loss"]
    methods = ["ru_regression", "joint_ru_regression"]
    datasets = ["mimic_los-density", "mimic_los-recip-density-sq"]
    for dataset in datasets:
        for loss in losses:
            for gamma in gammas:
                for method in methods:
                    exp_id = "{}_{}_{}_{}_1-9-24".format(dataset, method, loss, gamma)
                    script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                    with open(script_fn, "w") as f:
                        print(
                            SBATCH_PREFACE.format(
                                exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                            ),
                            file=f,
                        )

                        base_cmd = "python train.py main --dataset {}  --method {} --epochs 20 --seed {} --save mimic --gamma {}  --loss {} --model_class neural_network".format(
                            dataset, method, seed, gamma, loss
                        )
                        print(base_cmd, file=f)
                        print("sleep 1", file=f)


def generate_homoscedastic_example_runs():
    seed = 0
    gammas = [1.0, 2.0, 4.0, 8.0, 16.0]
    losses = ["squared_loss"]
    methods = ["joint_ru_regression", "ru_regression"]

    for loss in losses:
        for gamma in gammas:
            for method in methods:
                exp_id = "homoscedastic_example_{}_{}_{}_12-26-23".format(
                    method, loss, gamma
                )
                script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                        ),
                        file=f,
                    )

                    base_cmd = "python train.py main --dataset homoscedastic_one_dim --p_test_lo 0.5 --p_test_hi 0.5 --p_train 0.2 --n_test_sweep 1 --method {} --epochs 400 --loss squared_loss --seed {} --gamma {}".format(
                        method, seed, gamma
                    )
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)


def generate_heteroscedastic_example_runs():
    seed = 0
    gammas = [1.0, 2.0, 4.0, 8.0, 16.0]
    losses = ["squared_loss"]
    methods = ["joint_ru_regression", "ru_regression"]

    for loss in losses:
        for gamma in gammas:
            for method in methods:
                exp_id = "heteroscedastic_example_{}_{}_{}_12-26-23".format(
                    method, loss, gamma
                )
                script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                        ),
                        file=f,
                    )

                    base_cmd = "python train.py main --dataset heteroscedastic_one_dim --p_test_lo 0.5 --p_test_hi 0.5 --p_train 0.2 --n_test_sweep 1 --method {} --epochs 400 --loss squared_loss --seed {} --gamma {}".format(
                        method, seed, gamma
                    )
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)


#generate_severe_mimic_runs()
#generate_survey_runs()
#generate_survey_nn_runs()
generate_homoscedastic_example_runs()
generate_heteroscedastic_example_runs()
