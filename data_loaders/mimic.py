import pandas as pd
from data_loaders.dataloader_utils import get_dataloaders, standardize
import numpy as np
import torch
import os


def _load_mimic_los(unobserved):
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
    #    total_df['los'].where(total_df['los'] >= 7, 31)

    #    total_df['los'].where(total_df['los'] >= 10, 10)
    total_df = total_df[total_df["los"] <= 10]
    total_df = total_df[total_df["bp_diastolic"] <= 375.0]
    total_df = total_df[total_df["bp_systolic"] <= 375.0]
    total_df = total_df[total_df["o2sat"] <= 100.0]
    total_df = total_df[total_df["resp_rate"] <= 300.0]
    total_df = total_df[total_df["temp_fahren"] <= 113.0]

    # total_df["weight_avg_lbs"] = total_df["weight_avg_lbs"].where(total_df["weight_avg_lbs"] >= 250, 250)
    # total_df["fio2"] = total_df["fio2"].where(total_df["fio2"] >= 100, 100)
    total_df = total_df.fillna(impute_vals)
    total_df["gender"] = total_df["gender"].astype("category").cat.codes
    total_df["gender"] = total_df["gender"].replace(to_replace=0, value=2)
    total_df["ethnicity"] = total_df["ethnicity"].where(
        total_df["ethnicity"] == "[WHITE]", 2
    )
    total_df["ethnicity"] = total_df["ethnicity"].replace(to_replace="[WHITE]", value=1)

    # total_df['fio2'] = total_df['fio2'].apply(lambda x: x if x > 1 else 100 * x)
    # total_df = total_df[total_df["fio2"] <= 100.]

    #     update_vals = df.where(total_df["fio2"] < 1)
    #     total_df[update_vals]["fio2"] *= total_df[update_vals]["fio2"] * 100

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
    ]
    total_df = total_df.reset_index()
    return total_df[features], total_df[[unobserved]], total_df["los"]


def generate_weights(D_train, D_test, p, seed):
    D = (D_test - min(D_train)) / (max(D_train) - min(D_train))
    sample_weights = D ** (p)
    sample_weights /= sum(sample_weights)
    return torch.Tensor(sample_weights)


def get_mimic_dataloaders(
    seed, unobserved, p_train, p_test_lo, p_test_hi, n_test_sweep
):
    X, z, y = _load_mimic_los(unobserved)

    rng = np.random.RandomState(seed)
    permutation = rng.permutation(X.shape[0])
    index_train = permutation[: int(0.60 * X.shape[0])]
    index_test = permutation[int(0.60 * X.shape[0]) :]

    D_test = z.loc[index_test].to_numpy()
    D_train = z.loc[index_train].to_numpy()

    X_train = X.loc[index_train].to_numpy()
    y_train = y.loc[index_train].to_numpy()

    permutation = rng.permutation(X_train.shape[0])
    index_train = permutation[: int(0.60 * X_train.shape[0])]
    index_val = permutation[int(0.60 * X_train.shape[0]) :]
    X_train_new = X_train[index_train, :]
    X_val = X_train[index_val, :]
    y_train_new = y_train[index_train, None]
    y_val = y_train[index_val, None]

    if n_test_sweep == 1:
        p_tests = [p_test_lo]
    else:
        p_tests = list(np.linspace(p_test_lo, p_test_hi, n_test_sweep))

    test_weights = []
    X_test = X.loc[index_test].to_numpy()
    y_test = y.loc[index_test].to_numpy()
    y_test = y_test[:, None]
    X_tests = [X_test]
    y_tests = [y_test]

    print("train: ", len(X_train_new), "val: ", len(X_val), "test: ", len(X_test))

    for p_test in p_tests:
        sample_weights = generate_weights(D_train, D_test, p_test, seed=seed)
        test_weights.append(sample_weights)

    return (
        get_dataloaders(
            X_train_new,
            y_train_new,
            X_val,
            y_val,
            X_tests,
            y_tests,
            seed,
            batchsize=1024,
        ),
        p_tests,
        test_weights,
    )
