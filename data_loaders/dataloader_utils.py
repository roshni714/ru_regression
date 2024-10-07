import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset


def standardize(data):
    mu = data.mean(axis=0, keepdims=1)
    scale = data.std(axis=0, keepdims=1)
    # scale[scale < 1e-10] = 1.0
    data = (data - mu) / scale
    return data, mu, scale


def get_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test, r_test=None
):
    X_train, x_train_mu, x_train_scale = standardize(X_train)
    X_test = (X_test - x_train_mu) / x_train_scale
    X_val = (X_val - x_train_mu) / x_train_scale
    y_train, y_train_mu, y_train_scale = standardize(y_train)
    y_test = (y_test - y_train_mu) / y_train_scale
    y_val = (y_val - y_train_mu) / y_train_scale

    y_train_mu = y_train_mu.item()
    y_train_scale = y_train_scale.item()
    #    y_train_mu = 0.
    #    y_train_scale = 1.

    def helper(X, y):
        X = torch.Tensor(X)
        y = torch.Tensor(y).squeeze(dim=2)
        dataset = TensorDataset(X, y)
        return dataset

    train = helper(X_train, y_train)
    val = helper(X_val, y_val)
    test = helper(X_test, y_test)

        
    train_loader = DataLoader(
        train, batch_size=int(X_train.shape[0] * 0.25), shuffle=True
    )
    val_loader = DataLoader(val, batch_size=len(val), shuffle=False)
    test_loader = DataLoader(test, batch_size=len(test), shuffle=False)

    if r_test is not None:
        test2 = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test).squeeze(dim=2), torch.Tensor(r_test).squeeze(dim=2))
        test2_loader = DataLoader(test2, batch_size=len(test2), shuffle=False)
        return (
        train_loader,
        val_loader,
        [test_loader, test2_loader],
        X_train[0].shape[0],
        x_train_mu,
        x_train_scale,
        y_train_mu,
        y_train_scale,
    )

    else:
        return (
        train_loader,
        val_loader,
        [test_loader],
        X_train[0].shape[0],
        x_train_mu,
        x_train_scale,
        y_train_mu,
        y_train_scale)
