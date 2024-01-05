from data_loaders import (
    get_one_dim_dataloaders,
    get_shift_high_dim_dataloaders,
    get_mimic_dataloaders,
    get_survey_dataloaders,
)


def get_bound_function(gamma):
    def f(x, low=False, up=False):
        if low and not up:
            return 1 / gamma
        elif up and not low:
            return gamma
        else:
            assert (
                False
            ), "bound function received invalid arguments x={}, low={}, upper={}".format(
                x, low, up
            )

    return f


def get_dataset(
    dataset,
    n_train,
    d,
    p_train,
    p_test_lo,
    p_test_hi,
    n_test_sweep,
    use_train_weights,
    seed,
):
    if dataset == "heteroscedastic_one_dim" or dataset == "homoscedastic_one_dim":
        (
            train,
            val,
            test,
            input_size,
            X_mean,
            X_std,
            y_mean,
            y_std,
        ), p_tests = get_one_dim_dataloaders(
            dataset,
            n_train=n_train,
            p_train=p_train,
            p_test_lo=p_test_lo,
            p_test_hi=p_test_hi,
            n_test_sweep=n_test_sweep,
            seed=seed,
        )
        return train, val, test, input_size, X_mean, X_std, y_mean, y_std, p_tests, None

    elif dataset == "shifted_high_dim":
        (
            train,
            val,
            test,
            input_size,
            X_mean,
            X_std,
            y_mean,
            y_std,
        ), p_tests = get_shift_high_dim_dataloaders(
            dataset,
            n_train=n_train,
            d=d,
            p_train=p_train,
            p_test_lo=p_test_lo,
            p_test_hi=p_test_hi,
            n_test_sweep=n_test_sweep,
            seed=seed,
        )
        return train, val, test, input_size, X_mean, X_std, y_mean, y_std, p_tests, None
    elif "mimic" in dataset:
        (
            train,
            val,
            test,
            input_size,
            X_mean,
            X_std,
            y_mean,
            y_std,
        ) = get_mimic_dataloaders(dataset, seed=seed)
        return (train, val, test, input_size, X_mean, X_std, y_mean, y_std, None, None)
    elif dataset == "survey":
        (
            train,
            val,
            test,
            X_mean,
            X_std,
            y_mean,
            y_std,
            features,
        ) = get_survey_dataloaders("mental_health", use_train_weights, seed)
        input_size = len(features)

        return (train, val, test, input_size, X_mean, X_std, y_mean, y_std, None, None)
