import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset


def standardize(data):
    mu = data.mean(axis=0, keepdims=1)
    scale = data.std(axis=0, keepdims=1)
    # scale[scale < 1e-10] = 1.0
    data = (data - mu) / scale
    return data, mu, scale


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

    train_loader = DataLoader(
        train, batch_size=int(X_train.shape[0] * 0.25), shuffle=True
    )
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
