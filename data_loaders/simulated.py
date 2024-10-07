import torch
import numpy as np
from data_loaders.dataloader_utils import get_dataloaders


def generate_heteroscedastic_one_dim_dataset(n, p, seed):
    rng = np.random.RandomState(seed)
    us = rng.binomial(n=1, p=p, size=n)
    xs = rng.uniform(0.0, 10.0, size=n)
    zs = xs <= 6
    vs = rng.binomial(n=1, p=0.5, size=n)
    noise = rng.normal(0.0, 1.0, size=n)
    big_noise = (2 * rng.binomial(n=1, p=0.5, size=n) - 1) * 10
    ys = (
        zs * (np.sqrt(xs) + (np.sqrt(xs) * 3 + 1) * us)
        + (1 - zs) * 10 * (2 * vs - 1)
        + noise
    )

    return xs.reshape(-1, 1), ys.reshape(-1, 1, 1)


def generate_homoscedastic_one_dim_dataset(n, p, seed):
    rng = np.random.RandomState(seed)
    us = rng.binomial(n=1, p=p, size=n)
    xs = rng.uniform(0.0, 6.0, size=n)
    noise = rng.normal(0.0, 1.0, size=n)
    ys = np.sin(xs) + us * 5 + noise

    return xs.reshape(-1, 1), ys.reshape(-1, 1, 1)


def get_one_dim_dataloaders(
    dataset, n_train, seed, p_train, p_test_lo, p_test_hi, n_test_sweep
):
    if "homoscedastic" in dataset:
        generate_data = generate_homoscedastic_one_dim_dataset
    else:
        generate_data = generate_heteroscedastic_one_dim_dataset

    X_val, y_val = generate_data(n=int(n_train * 0.2), p=p_train, seed=seed)

    if n_test_sweep == 1:
        p_tests = [p_test_lo]
    elif n_test_sweep == 5:
        p_tests = [0.1, 0.2, 0.5, 0.7, 0.9]
    else:
        p_tests = np.linspace(p_test_lo, p_test_hi, n_test_sweep)
    X_tests = []
    y_tests = []
    for p_test in p_tests:
        X_test, y_test = generate_data(n=20000, p=p_test, seed=seed + 1)
        X_tests.append(X_test)
        y_tests.append(y_test)

    X_train, y_train = generate_data(n=n_train, p=p_train, seed=seed + 2)
    return (
        get_dataloaders(X_train, y_train, X_val, y_val, X_tests, y_tests),
        p_tests,
    )


def generate_shift_high_dim_dataset(n, d, p, seed):
    """
    Simple hyperplane experiment
    """
    rng = np.random.RandomState(seed)
    us = rng.binomial(n=1, p=p, size=n).reshape(n, 1, 1)
    xs = rng.uniform(0.0, 1.0, size=(n * d)).reshape(n, d, 1)
    noise = rng.normal(0.0, 0.05, size=n).reshape(n, 1, 1)
    rng2 = np.random.RandomState(0)
    beta = rng2.uniform(-1.0, 1.0, size=d).reshape(d, 1)
    ys = np.matmul(beta.T, xs) + us * 0.5 + noise
    return xs.reshape(n, d), ys.reshape(n, 1), us, beta


def sample_y_conditional_shift_one_dim(n, p, x, seed):
    rng = np.random.RandomState(seed)
    us = rng.binomial(n=1, p=p, size=n)
    noise = rng.normal(0.0, 1.0, size=n)
    ys = np.sqrt(x) + (np.sqrt(x) * 3 + 1) * us + noise
    return ys


def get_shift_high_dim_dataloaders(
    dataset, n_train, d, seed, p_train, p_test_lo, p_test_hi, n_test_sweep
):
    X_val, y_val, _, beta = generate_shift_high_dim_dataset(
        n=int(n_train * 0.2), d=d, p=p_train, seed=seed
    )
    if n_test_sweep == 5:
        p_tests = [0.1, 0.2, 0.5, 0.7, 0.9]
    if n_test_sweep == 1:
        p_tests = [p_test_lo]
    X_tests = []
    y_tests = []
    for p_test in p_tests:
        X_test, y_test, _, beta1 = generate_shift_high_dim_dataset(
            n=20000, d=d, p=p_test, seed=seed + 1
        )
        np.testing.assert_allclose(beta, beta1)
        X_tests.append(X_test)
        y_tests.append(y_test)

    X_train, y_train, _, _ = generate_shift_high_dim_dataset(
        n=n_train, d=d, p=p_train, seed=seed + 2
    )
    print("Beta: {}".format(beta.flatten()))
    return (
        get_dataloaders(X_train, y_train, X_val, y_val, X_tests, y_tests, seed),
        p_tests,
    )
