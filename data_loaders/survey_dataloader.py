import pandas as pd
import numpy as np
import torch
import os
import unittest
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset


np.random.seed(0)


def standardize(data):
    mu = data.mean(axis=0, keepdims=1)
    scale = data.std(axis=0, keepdims=1) + 1e-5
    data = (data - mu) / scale
    return data, mu, scale


def _load_data(dataset, outcome=None):
    if "brfss" in dataset:
        df = pd.read_csv("/home/groups/swager/mreitsma/{}_num.csv".format(dataset))
        if outcome is not None:
            df = df.dropna(axis=0, subset=outcome).reset_index()

        remove_states = set(
            [
                "American Samoa",
                "Guam",
                "Northern Mariana Islands",
                "Puerto Rico",
                "U.S. Virgin Islands",
            ]
        )
        df = df[~df["state_name"].isin(remove_states)].reset_index(drop=True)
        for race_eth in ["NH AIAN", "NH Multiple", "NH NHPI"]:
            df.loc[df["race_eth"] == race_eth, "race_eth"] = "NH Other"

        df["state_name"] = df["state_name"].astype("category")
        df["state_name"] = df["state_name"].cat.add_categories("Florida")
        df["marital"] = df["marital"].astype("category")
        one_hot_state = pd.get_dummies(df[["state_name"]])
        one_hot_race_eth = pd.get_dummies(df[["race_eth"]])
        one_hot_marital = pd.get_dummies(df[["marital"]])
        features = [
            "age_grp",
            "male",
            "edu_cat",
            "income_detailed",
            "any_ins",
            "emp_ins",
            "medicare_ins",
            "medicaid_ins",
            "other_ins",
            "household_size",
        ]
        weights = df[["LLCPWT"]]

    elif "hps" in dataset:
        df = pd.read_csv("/home/groups/swager/mreitsma/hps_prepped_date_novaxcat.csv")
        df["mental_health"] = df["sum_mh"]
        #       df[["ANXIOUS", "DOWN", "WORRY", "INTEREST"]].max(axis=1)
        #       mapping = {1: 0, 2: 7, 3: 15, 4: 30}
        #       df["mental_health"] = df["mental_health"].map(mapping)

        if outcome is not None:
            df = df.dropna(axis=0, subset=outcome).reset_index()

        remove_states = set(
            [
                "American Samoa",
                "Guam",
                "Northern Mariana Islands",
                "Puerto Rico",
                "U.S. Virgin Islands",
            ]
        )
        df = df[~df["state_name"].isin(remove_states)].reset_index(drop=True)
        year = dataset.split("_")[-1]
        df = df[df.YEAR == int(year)].reset_index()

        features = [
            "age_grp",
            "male",
            "edu_cat",
            "income_detailed",
            "any_ins",
            "emp_ins",
            "medicare_ins",
            "medicaid_ins",
            "other_ins",
            "household_size",
        ]

        one_hot_state = pd.get_dummies(df[["state_name"]])
        one_hot_race_eth = pd.get_dummies(df[["race_eth"]])
        one_hot_marital = pd.get_dummies(df[["marital"]])

        weights = df[["PWEIGHT"]]

    X = df[features]
    X = X.join(one_hot_state)
    X = X.join(one_hot_race_eth)
    X = X.join(one_hot_marital)
    X.fillna(X.mean(), inplace=True)
    feat = sorted(list(X.columns))
    X = X[feat]

    if outcome is not None:
        y = df[[outcome]]
        #        return X[:1000], y[:1000], weights[:1000], X.columns
        return X, y, weights, X.columns
    else:
        return X, weights, X.columns


def get_survey_dataloaders(outcome, use_train_weights, seed):
    X, y, r, features = _load_data("hps_2021", outcome)
    X_test, y_test, r_test, test_features = _load_data("brfss_2021", outcome)
    assert sum(features == test_features) == len(features)
    y_test = (y_test / 30) * (12)
    y -= 4
    #    y /= 16.
    #    import pdb
    #    pdb.set_trace()
    #    X_test = X.copy()
    #    y_test = y.copy()
    #    r_test = r.copy()

    r /= r.sum()
    r_test /= r_test.sum()

    rng = np.random.RandomState(seed)
    permutation = rng.permutation(X.shape[0])
    index_train = permutation[: int(0.60 * X.shape[0])]
    index_val = permutation[int(0.60 * X.shape[0]) :]
    X_train = X.loc[index_train].to_numpy()
    y_train = y.loc[index_train].to_numpy()
    r_train = r.loc[index_train].to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    r_test = r_test.to_numpy()
    X_val = X.loc[index_val].to_numpy()
    y_val = y.loc[index_val].to_numpy()
    r_val = r.loc[index_val].to_numpy()

    y_test = y_test[:, None]
    r_test = r_test[:, None]
    y_train = y_train[:, None]
    r_train = r_train[:, None]
    r_val = r_val[:, None]
    y_val = y_val[:, None]

    print("train: ", len(X_train), "val: ", len(X_val), "test: ", len(X_test))

    X_train, x_train_mu, x_train_scale = standardize(X_train)
    X_val = (X_val - x_train_mu) / x_train_scale
    X_test = (X_test - x_train_mu) / x_train_scale

    if outcome in ["bmi"]:
        y_train, y_train_mu, y_train_scale = standardize(y_train)
        y_val = (y_val - y_train_mu) / y_train_scale
        y_train_mu = y_train_mu.item()
        y_train_scale = y_train_scale.item()
        y_test = (y_test - y_train_mu) / y_train_scale
    else:
        y_train_mu = 0
        y_train_scale = 1

    train_loaders, val_loaders, test_loader = format_dataloaders(
        X_train,
        y_train,
        r_train,
        X_val,
        y_val,
        r_val,
        X_test,
        y_test,
        r_test,
        use_train_weights,
        seed,
    )
    return (
        train_loaders,
        val_loaders,
        test_loader,
        x_train_mu,
        x_train_scale,
        y_train_mu,
        y_train_scale,
        features,
    )


def format_dataloaders(
    X_train,
    y_train,
    r_train,
    X_val,
    y_val,
    r_val,
    X_test,
    y_test,
    r_test,
    use_train_weights,
    seed=0,
):
    rng = np.random.RandomState(seed + 2)
    permutation = rng.permutation(X_train.shape[0])
    X_train = X_train[permutation, :]
    y_train = y_train[permutation]
    r_train = r_train[permutation]

    train_lengths = [len(X_train)]

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train).squeeze(dim=2)
    r_train = torch.Tensor(r_train).squeeze(dim=2)

    if use_train_weights:
        train = TensorDataset(X_train, y_train, r_train)
        val = TensorDataset(
            torch.Tensor(X_val),
            torch.Tensor(y_val).squeeze(dim=2),
            torch.Tensor(r_val).squeeze(dim=2),
        )

    else:
        train = TensorDataset(X_train, y_train)
        val = TensorDataset(
            torch.Tensor(X_val),
            torch.Tensor(y_val).squeeze(dim=2),
        )

    train_loader = DataLoader(train, batch_size=10000, shuffle=True)

    val_loader = DataLoader(val, batch_size=len(val), shuffle=False)
    test = TensorDataset(
        torch.Tensor(X_test),
        torch.Tensor(y_test).squeeze(dim=2),
        torch.Tensor(r_test).squeeze(dim=2),
    )
    test_loader = DataLoader(test, batch_size=len(test), shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
    )
