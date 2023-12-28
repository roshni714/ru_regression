import torch
import numpy as np
from data_loaders.dataloader_utils import get_dataloaders

def generate_joint_shift_one_dim_dataset(n, p, lamb, seed):
    rng = np.random.RandomState(seed)
    us = rng.binomial(n=1, p=p, size=n)
    zs = rng.binomial(n=1, p=lamb, size=n)
    n_low = np.sum(zs)
    x1s = rng.uniform(0, 5, size= n_low)
    x2s = rng.uniform(5, 10, size=n-n_low)
    xs = np.concatenate((x1s, x2s))
    noise = rng.normal(0.0, 1.0, size=n)
    ys = np.sqrt(xs) + (np.sqrt(xs) * 3 + 1) * us + noise
    return xs.reshape(-1, 1), ys.reshape(-1, 1) 

def get_joint_shift_one_dim_dataloaders(
    dataset, n_train, seed, p_train, p_test, lamb_test
):
    X_val, y_val = generate_joint_shift_one_dim_dataset(
        n=int(n_train * 0.2), p=p_train, lamb=0.25, seed=seed
    )
    X_tests = []
    y_tests = []

    X_test, y_test  = generate_joint_shift_one_dim_dataset(
            n=20000, p=p_test, lamb=lamb_test, seed=seed + 1)
    X_tests.append(X_test)
    y_tests.append(y_test)

    X_train, y_train = generate_joint_shift_one_dim_dataset(
        n=n_train, p=p_train, lamb=0.25, seed=seed + 2
    )
    return get_dataloaders(X_train, y_train, X_val, y_val, X_tests, y_tests, seed), [p_test]
