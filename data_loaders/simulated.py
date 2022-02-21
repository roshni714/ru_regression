import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

def standardize(data):
    mu = data.mean(axis=0, keepdims=1)
    scale = data.std(axis=0, keepdims=1)
    scale[scale < 1e-10] = 1.0
    data = (data - mu) / scale
    return data, mu, scale


def generate_shift_dataset(n, p_d1):
    n_d1 = int(n * p_d1)
    n_d2 = n - n_d1
    xs_d1 = np.random.uniform(0., 10., n_d1)
    ys_d1 = np.sqrt(xs_d1) + np.random.normal(0., 1., n_d1)
    xs_d2 = np.random.uniform(0., 10., n_d2)
    ys_d2 = 4 *np.sqrt(xs_d2) + 1 + np.random.normal(0., 1., n_d2)
    
    xs = np.hstack((xs_d1, xs_d2))
    ys = np.hstack((ys_d1, ys_d2))
    
    return xs, ys


def get_shift_dataloaders(dataset, seed, batch_size):
    X_train, y_train = generate_shift_dataset(n=7000, p_d1=0.8)
    X_val, y_val = generate_shift_dataset(n=2000, p_d1= 0.5)
    X_test, y_test = generate_shift_dataset(n=1000, p_d1=0.5)

    X_train, x_train_mu, x_train_scale = standardize(X_train)
    X_test = (X_test - x_train_mu) / x_train_scale
    y_train, y_train_mu, y_train_scale = standardize(y_train)
    y_test = (y_test - y_train_mu) / y_train_scale
    X_val = (X_val - x_train_mu) / x_train_scale
    y_val = (y_val - y_train_mu) / y_train_scale

    train = TensorDataset(
        torch.Tensor(X_train),
        torch.Tensor(y_train),
    )

    val = TensorDataset(
        torch.Tensor(X_val),
        torch.Tensor(y_val),
    )

    test = TensorDataset(
        torch.Tensor(X_test),
        torch.Tensor(y_test),
    )

    train_loader = DataLoader(train, batch_size=len(train), shuffle=True)
    val_loader = DataLoader(val, batch_size=len(val), shuffle=False)
    test_loader = DataLoader(test, batch_size=len(test), shuffle=False)
    return (
        train_loader,
        val_loader,
        test_loader,
        X[0].shape[0],
    )


