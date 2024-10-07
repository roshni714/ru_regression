import pandas as pd
import numpy as np
import torch
import os
import unittest
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

np.random.seed(0)

CONVERSION_DATA_PATH="/home/groups/swager/rsahoo/brfss_mental_health.csv"
CENSUS_DATA_PATH= "/home/groups/swager/rsahoo/acs_state_characteristics.csv"
BRFSS_DATA_PATH="/home/groups/swager/mreitsma/brfss_2021_num.csv"
HPS_DATA_PATH="/home/groups/swager/mreitsma/hps_prepped_date_novaxcat.csv"

def standardize(data):
    mu = data.mean(axis=0, keepdims=1)
    scale = data.std(axis=0, keepdims=1) + 1e-5
    data = (data - mu) / scale
    return data, mu, scale


def learn_conversion_30d_to_phq4():
    df = pd.read_csv(CONVERSION_DATA_PATH)
    df["PHQ_4"] = df["ADPLEAS1"] + df["ADDOWN1"] + df["FEELNERV"] + df["STOPWORY"]
    y = df["PHQ_4"].to_numpy().flatten() - 4.0
    X = df["MENTHLTH"].to_numpy().reshape(len(df), 1)
    model = IsotonicRegression().fit(X, y)
    z = np.linspace(min(X), max(X), 1000).reshape(1000, 1)
    w = model.predict(z)
    inv_model = interp1d(
        z.flatten(), w.flatten(), bounds_error=False, kind="previous", fill_value=(min(w), max(w))
    )
    return inv_model

def _load_data(dataset, outcome=None, reduced_feature_set=False):

    df_state_char = pd.read_csv(CENSUS_DATA_PATH)
    
    if "brfss" in dataset:
        df = pd.read_csv(BRFSS_DATA_PATH)
        if outcome is not None:
            df = df.dropna(axis=0, subset=outcome).reset_index()

        model = learn_conversion_30d_to_phq4()
        df["mental_health"] = model(df["mental_health"])

        remove_states = set(
            [
                "American Samoa",
                "Guam",
                "Northern Mariana Islands",
                "Puerto Rico",
                "U.S. Virgin Islands",
                "Florida"
            ]
        )
        df = df[~df["state_name"].isin(remove_states)].reset_index(drop=True)
        for race_eth in ["NH AIAN", "NH Multiple", "NH NHPI"]:
            df.loc[df["race_eth"] == race_eth, "race_eth"] = "NH Other"

        df["state_name"] = df["state_name"].astype("category")        
        df["marital"] = df["marital"].astype("category")
        one_hot_race_eth = pd.get_dummies(df[["race_eth"]])
        one_hot_marital = pd.get_dummies(df[["marital"]])

        df = pd.merge(df, df_state_char, on="state_name", how="left")
    elif "hps" in dataset:
        df = pd.read_csv(HPS_DATA_PATH)
        df["mental_health"] = df["sum_mh"] - 4.0

        if outcome is not None:
            df = df.dropna(axis=0, subset=outcome).reset_index()

        remove_states = set(
            [
                "American Samoa",
                "Guam",
                "Northern Mariana Islands",
                "Puerto Rico",
                "U.S. Virgin Islands",
                "Florida"
            ]
        )
        df = df[~df["state_name"].isin(remove_states)].reset_index(drop=True)
        year = dataset.split("_")[-1]
        df = df[df.YEAR == int(year)].reset_index()

        one_hot_race_eth = pd.get_dummies(df[["race_eth"]])
        one_hot_marital = pd.get_dummies(df[["marital"]])
        df = pd.merge(df, df_state_char, on="state_name", how="left")

    if reduced_feature_set:
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
        X = df[features]
        X.fillna(X.mean(), inplace=True)
        feat = sorted(list(X.columns))
        X = X[feat]

    else:
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
            '4yr_college',
            'any_health_insurance', 
            'avg_hh_size', 
            'edu_hs_or_less', 
            'english_only',
            'female_never_married', 
            'fertility_rate', 
            'food_stamps',
            'graduate_degree', 
            'hh_computer', 
            'hh_internet', 
            'male_never_married',
            'mean_income', 
            'median_house_value', 
            'median_income', 
            'median_rent',
            'poverty', 
            'private_health_insurance', 
            'some_college_or_2yr',
            'unemployment_rate', 
            'us_born', 
            'veterans', 
            'republican_pct',
            'tot_pop'
        ]

        X = df[features]
        X = X.join(one_hot_race_eth)
        X = X.join(one_hot_marital)
        X.fillna(X.mean(), inplace=True)
        feat = sorted(list(X.columns))
        X = X[feat]
        
    if outcome is not None:
        y = df[[outcome]]
        #        return X[:40000], y[:40000], weights[:40000], X.columns
        return X, y, X.columns
    else:
        return X, X.columns


def learn_train_weights(X0, X1, weight_mask):
    rng = np.random.RandomState(30)

    #    if len(X0) > len(X1):
    #        permutation = rng.permutation(X0.shape[0])

    #        dist0_X = X0[permutation[:len(X1)], ]
    #        dist1_X = X1
    #    else:
    #        permutation = rng.permutation(X1.shape[0])
    #        dist0_X = X0
    #        dist1_X = X1[permutation[:len(X0)],]

    dist0_X = X0
    dist1_X = X1
    dist0_y = np.zeros(dist0_X.shape[0])
    dist1_y = np.ones(dist1_X.shape[0])

    all_X = np.vstack((dist0_X, dist1_X))
    all_y = np.hstack((dist0_y, dist1_y))

    clf = DecisionTreeClassifier(class_weight="balanced").fit(
        all_X[:, weight_mask], all_y
    )

    def density_ratio_function(X):
        ratio = clf.predict_proba(X[:, weight_mask])[:, 1] / np.clip(
            clf.predict_proba(X[:, weight_mask])[:, 0], a_min=1e-2, a_max=1.0
        )
        #         import pdb
        #         pdb.set_trace()

        return ratio.reshape(X.shape[0], 1, 1)

    return density_ratio_function


def get_survey_dataloaders(reduced_feature_set, use_train_weights, seed, y_normalize=False):
    X, y, features = _load_data("hps_2021", "mental_health", reduced_feature_set)
    X_test, y_test, test_features = _load_data("brfss_2021", "mental_health", reduced_feature_set)

    weight_mask = []
    for feat in features:
        if (
            feat in ["age_grp", "male", "edu_cat", "income_detailed"]
            or "race_eth" in feat
        ):
            weight_mask.append(True)
        else:
            weight_mask.append(False)
    assert sum(features == test_features) == len(features)

    rng = np.random.RandomState(seed)
    permutation = rng.permutation(X.shape[0])
    index_train_and_val = permutation[: int(0.60 * X.shape[0])]
    index_test2 = permutation[int(0.60 * X.shape[0]) :]
    index_train = index_train_and_val[: int(0.60 * len(index_train_and_val))]
    index_val = index_train_and_val[int(0.60 * len(index_train_and_val)) :]
    X_train = X.loc[index_train].to_numpy()
    y_train = y.loc[index_train].to_numpy()

    X_test2 = X.loc[index_test2].to_numpy()
    y_test2 = y.loc[index_test2].to_numpy()

    X_val = X.loc[index_val].to_numpy()
    y_val = y.loc[index_val].to_numpy()

    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    y_test = y_test[:, None]
    y_train = y_train[:, None]
    y_val = y_val[:, None]
    y_test2 = y_test2[:, None]
    
    print(
        "train: ",
        len(X_train),
        "val: ",
        len(X_val),
        "test: ",
        len(X_test),
        "test2:",
        len(X_test2),
    )

    X_train, x_train_mu, x_train_scale = standardize(X_train)
    X_val = (X_val - x_train_mu) / x_train_scale
    X_test = (X_test - x_train_mu) / x_train_scale
    X_test2 = (X_test2 - x_train_mu) / x_train_scale

    train_X = np.vstack((X_train, X_val))
    density_ratio_function = learn_train_weights(train_X, X_test, weight_mask)
    r_test2 = None

    if use_train_weights:
        r_train = density_ratio_function(X_train)
        r_val = density_ratio_function(X_val)
    else:
        r_val = None
        r_train = None
        
    if y_normalize:
       y_train, y_train_mu, y_train_scale = standardize(y_train)
       y_val = (y_val - y_train_mu)/y_train_scale
       y_test = (y_test - y_train_mu)/y_train_scale
       y_test2 = (y_test2 - y_train_mu)/y_train_scale
       y_train_mu = y_train_mu.item()
       y_train_scale = y_train_scale.item()
    else:
        y_train_mu = 0
        y_train_scale = 1

    train_loader, val_loader, test_loaders = format_dataloaders(
        X_train=X_train,
        y_train=y_train,
        r_train=r_train,
        X_val=X_val,
        y_val=y_val,
        r_val=r_val,
        X_test=X_test,
        y_test=y_test,
        r_test=None,
        X_test2=X_test2,
        y_test2=y_test2,
        r_test2=r_test2,
    )

    return (
        train_loader,
        val_loader,
        test_loaders,
        x_train_mu,
        x_train_scale,
        y_train_mu,
        y_train_scale,
        features,
    )


def format_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    X_test2,
    y_test2,
    r_train=None,
    r_val=None,
    r_test=None,
    r_test2=None,
):
    def helper(X, y, r):
        X = torch.Tensor(X)
        y = torch.Tensor(y).squeeze(dim=2)
        if r is not None:
            r = torch.Tensor(r).squeeze(dim=2)
            dataset = TensorDataset(X, y, r)
        else:
            dataset = TensorDataset(X, y)
        return dataset

    train = helper(X_train, y_train, r_train)
    val = helper(X_val, y_val, r_val)
    test = helper(X_test, y_test, r_test)
    test2 = helper(X_test2, y_test2, r_test2)

    train_loader = DataLoader(train, batch_size=20000, shuffle=True)
    val_loader = DataLoader(val, batch_size=len(val), shuffle=False)
    test_loader = DataLoader(test, batch_size=len(test), shuffle=False)
    test2_loader = DataLoader(test2, batch_size=len(test2), shuffle=False)

    return (
        train_loader,
        val_loader,
        [test_loader, test2_loader],
    )
