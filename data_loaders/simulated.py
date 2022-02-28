import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset


def standardize(data):
    mu = data.mean(axis=0, keepdims=1)
    scale = data.std(axis=0, keepdims=1)
    scale[scale < 1e-10] = 1.0
    data = (data - mu) / scale
    return data, mu, scale


def generate_shift_dataset(n, p, seed):
    rng = np.random.RandomState(seed)
    us = rng.binomial(n=1, p=p, size=n)
    xs = rng.uniform(0.0, 10.0, size=n)
    noise = rng.normal(0.0, 1.0, size=n)
    ys = np.sqrt(xs) + (np.sqrt(xs) * 3 + 1) * us + noise

    return xs.reshape(-1, 1), ys.reshape(-1, 1)


def get_shift_dataloaders(dataset, seed, p_train, p_test_lo, p_test_hi, n_test_sweep):
    X_val, y_val = generate_shift_dataset(n=2000, p=p_train, seed=seed)

    p_tests = np.linspace(p_test_lo, p_test_hi, n_test_sweep)
    if n_test_sweep == 1:
        p_tests = [p_test_lo]
    X_tests = []
    y_tests = []
    for p_test in p_tests:
        X_test, y_test = generate_shift_dataset(n=1000, p=p_test, seed=seed + 1)
        X_tests.append(X_test)
        y_tests.append(y_test)

    X_train, y_train = generate_shift_dataset(n=7000, p=p_train, seed=seed + 2)
    return (
        get_dataloaders(X_train, y_train, X_val, y_val, X_tests, y_tests, seed),
        p_tests,
    )


# def get_shift_oracle_dataloaders(dataset, seed, p_train):
#    X_val, y_val = generate_shift_dataset(n=2000, p=p_train, seed=seed)
#    X_test, y_test = generate_shift_dataset(n=1000, p=p_train, seed=seed + 1)
#    X_train, y_train = generate_shift_dataset(n=7000, p=p_train, seed=seed + 2)
#    return get_dataloaders(X_train, y_train, X_val, y_val, [X_test], [y_test], seed)


def get_dataloaders(X_train, y_train, X_val, y_val, X_tests, y_tests, seed):
    X_train, x_train_mu, x_train_scale = standardize(X_train)
    y_train, y_train_mu, y_train_scale = standardize(y_train)
    for i in range(len(y_tests)):
        y_tests[i] = (y_tests[i] - y_train_mu) / y_train_scale
        X_tests[i] = (X_tests[i] - x_train_mu) / x_train_scale

    X_val = (X_val - x_train_mu) / x_train_scale
    y_val = (y_val - y_train_mu) / y_train_scale

    y_train_mu = y_train_mu.item()
    y_train_scale = y_train_scale.item()
    train = TensorDataset(
        torch.Tensor(X_train),
        torch.Tensor(y_train),
    )

    val = TensorDataset(
        torch.Tensor(X_val),
        torch.Tensor(y_val),
    )

    test_loaders = []
    for i in range(len(y_tests)):
        test = TensorDataset(
            torch.Tensor(X_tests[i]),
            torch.Tensor(y_tests[i]),
        )
        test_loader = DataLoader(test, batch_size=len(test), shuffle=False)
        test_loaders.append(test_loader)

    train_loader = DataLoader(train, batch_size=1000, shuffle=True)
    val_loader = DataLoader(val, batch_size=len(val), shuffle=False)
    return (
        train_loader,
        val_loader,
        test_loaders,
        X_train[0].shape[0],
        x_train_mu,
        x_train_scale,
        y_train_mu,
        y_train_scale,
    )
