import pandas as pd
from dataloader_utils import get_dataloaders
import numpy as np
import torch
import os


def _load_mimic_los():
    df = pd.read_excel("data_loaders/data/mimic/mimic_050221.xlsx")
    df2 = pd.read_excel("data_loaders/data/mimic/mimic_050221_2.xlsx")
    total_df = pd.concat([df, df2])
    impute_vals = {
        "cap_refill": 0.0,
        "bp_diastolic": 59.0,
        "fio2": 0.21,
        "gcs_eye": 4,
        "gcs_motor": 6,
        "gcs_total": 15,
        "gcs_verbal": 5,
        "glucose": 128.0,
        "heart_rate": 86.0,
        "height_cm": 170,
        "bp_mean": 77.0,
        "o2sat": 98.0,
        "resp_rate": 19,
        "bp_systolic": 118.0,
        "temp_fahren": 97.88,
        "weight_lbs": 81.0,
    }
    total_df = total_df[total_df["age_on_adm"] < 300]
    total_df = total_df.fillna(impute_vals)

    features = [
        "cap_refill",
        "bp_diastolic",
        "bp_systolic",
        "bp_mean",
        "fio2",
        "gcs_eye",
        "gcs_verbal",
        "gcs_motor",
        "gcs_total",
        "glucose",
        "heart_rate",
        "height_cm",
        "o2sat",
        "resp_rate",
        "temp_fahren",
        "weight_lbs",
        "age_on_adm",
    ]
    return total_df[features].to_numpy(), total_df["los"].to_numpy()


def sample_dataset(D, X, y, median_age, n, p, seed):
    rng = np.random.RandomState(seed)

    D_0 = D.index[D["age_on_adm" <= median_age]].tolist()
    D_1 = D.index[D["age_on_adm" > median_age]].tolist()

    us = rng.binomial(n=1, p=p_train, size=n_train).reshape(n, 1, 1)
    n_0 = np.sum(us == 1)
    n_1 = np.sum(us == 0)

    l_idx_0 = rng.choice(n_0, replace=False)
    l_idx_1 = rng.choice(n_1, replace=False)
    idx_0 = D_0[l_idx_0]
    idx_1 = D_1[l_idx_1]

    X_0 = X[idx_0, :]
    X_1 = X[idx_1, :]
    y_0 = y[idx_0, None]
    y_1 = y[idx_1, None]

    X_sampled = pd.concat([X_0, X_1], ignore_index=True)
    y_sampled = pd.concat([y_0, y_1], ignore_index=True)

    return X_sampled, y_sampled


def get_mimic_dataloaders(
    n_train, d, seed, p_train, p_test_lo, p_test_hi, n_test_sweep
):
    X, y = _load_mimic_los()
    age_col = X["age_on_adm"]
    median_age = X["age_on_adm"].median()

    rng = np.random.RandomState(seed)
    permutation = rng.permutation(X.shape[0])
    index_test = permutation[: int(2 * X.shape[0] / 3)]
    index_train = permuation[int(2 * X.shape[0] / 3) :]

    D_test = X[index_test, :]
    D_train = X[index_train, :]

    X_train, y_train = sample_dataset(D_train, X, y, median_age, n, p_train, seed=seed)
    X_train_new = X_train[index_train, :]
    X_val = X_train[index_val, :]
    y_train_new = y_train[index_train, None]
    y_val = y_train[index_val, None]

    if n_test_sweep == 5:
        p_tests = [0.1, 0.2, 0.5, 0.7, 0.9]
    if n_test_sweep == 1:
        p_tests = [p_test_lo]
    X_tests = []
    y_tests = []
    for p_test in p_tests:
        X_test, y_test = sample_dataset(D_test, X, y, median_age, n, p_test, seed=seed)
        X_tests.append(X_test)
        y_tests.append(y_test)

    permutation = rng.permutation(X_train.shape[0])
    index_train = permutation[: int(3 * X.shape[0] / 4)]
    index_val = permutation[int(3 * X.shape[0] / 4) :]

    return (
        get_dataloaders(X_train_new, y_train_new, X_val, y_val, X_tests, y_tests, seed),
        p_tests,
    )


if __name__ == "__main__":
    print(_load_mimic_los()[0].shape)
