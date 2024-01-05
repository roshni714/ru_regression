import pandas as pd
from data_loaders.dataloader_utils import get_dataloaders, standardize
import numpy as np
import torch
import os

np.random.seed(100)


def _load_mimic_los(dataset):
    l = dataset.split("_")
    if len(l) == 1:
        weight_type = None
    else:
        weight_type = l[1]
    df = pd.read_csv("data_loaders/data/mimic_2022/mimic_051522_1.csv")
    df2 = pd.read_csv("data_loaders/data/mimic_2022/mimic_051522_2.csv")
    total_df = pd.concat([df, df2], ignore_index=True)
    impute_vals = {
        "cap_refill": 0.0,
        "bp_diastolic": 59.0,
        "fio2": 21.0,
        "gcs_eye": 4,
        "gcs_motor": 6,
        "gcs_total": 15,
        "gcs_verbal": 5,
        "glucose": 128.0,
        "heart_rate": 86.0,
        "height_avg_cm": 170,
        "bp_mean": 77.0,
        "o2sat": 98.0,
        "resp_rate": 19,
        "bp_systolic": 118.0,
        "temp_fahren": 97.88,
        "weight_avg_lbs": 81.0,
        "ph": 7.4,
    }

    total_df = total_df[total_df["age_on_adm"] < 300.0]
    total_df = total_df[total_df["los"] <= 14]
    total_df = total_df[total_df["bp_diastolic"] <= 375.0]
    total_df = total_df[total_df["bp_systolic"] <= 375.0]
    total_df = total_df[total_df["o2sat"] <= 100.0]
    total_df = total_df[total_df["resp_rate"] <= 300.0]
    total_df = total_df[total_df["temp_fahren"] <= 113.0]
    total_df = total_df.fillna(impute_vals)

    total_df["gender"] = total_df["gender"].map({"M": 1.0, "F": 0.0})
    total_df["ethnicity"] = total_df["ethnicity"].apply(
        lambda x: 0.0 if x != "[WHITE]" else 1.0
    )
    total_df = total_df.reset_index()

    r = compute_weights(total_df, weight_type)
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
        "height_avg_cm",
        "o2sat",
        "resp_rate",
        "temp_fahren",
        "weight_avg_lbs",
        "ph",
        "gender",
    ]
    return total_df[features], total_df["los"], r


def compute_weights(total_df, weight_type):
    if weight_type == "los":
        r = 1 / total_df["los"].to_numpy()
    elif weight_type == "age":
        r = 1 / total_df["age_on_adm"].to_numpy()
    elif weight_type == "gender":
        r = total_df["gender"]
    elif weight_type == "ethnicity":
        r = total_df["ethnicity"]
    elif weight_type == None:
        r = np.ones(total_df["age_on_adm"].to_numpy().shape)
    return r


def get_mimic_dataloaders(dataset, seed):
    X, y, r = _load_mimic_los(dataset)

    rng = np.random.RandomState(1000)
    permutation = rng.permutation(X.shape[0])
    index_train = permutation[: int(0.60 * X.shape[0])]
    index_test = permutation[int(0.60 * X.shape[0]) :]

    X_train = X.loc[index_train].to_numpy()
    y_train = y.loc[index_train].to_numpy().reshape(len(index_train), 1, 1)
    r_train = r[index_train, None, None]

    X_test = X.loc[index_test].to_numpy()
    y_test = y.loc[index_test].to_numpy().reshape(len(index_test), 1, 1)
    r_test = r[index_test, None, None]

    permutation = rng.permutation(X_train.shape[0])
    index_train = permutation[: int(0.60 * X_train.shape[0])]
    index_val = permutation[int(0.60 * X_train.shape[0]) :]
    X_train_new = X_train[index_train, :]
    X_val = X_train[index_val, :]
    y_train_new = y_train[index_train, :, :]
    y_val = y_train[index_val, :, :]
    r_train_new = r_train[index_train, :, :]
    r_val = r_train[index_val, :, :]

    def generate_data(X, y, r, n):
        idx = np.random.choice(len(y), size=n, p=(r / r.sum()).flatten())
        X_new = X[idx, :]
        y_new = y[idx, :]
        return X_new, y_new

    X_train_new, y_train_new = generate_data(
        X_train_new, y_train_new, r_train_new, n=10000
    )
    X_val, y_val = generate_data(X_val, y_val, r_val, n=4000)
    X_test_biased, y_test_biased = generate_data(X_test, y_test, r_test, n=len(X_test))

    print("train: ", len(X_train_new), "val: ", len(X_val), "test: ", len(X_test))

    return get_dataloaders(
        X_train_new,
        y_train_new,
        X_val,
        y_val,
        X_test,
        y_test,
        X_test_biased,
        y_test_biased,
    )
