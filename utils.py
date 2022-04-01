from data_loaders import (
    get_shift_one_dim_dataloaders,
    get_shift_high_dim_dataloaders,
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


def get_dataset(dataset, n_train, d, p_train, p_test_lo, p_test_hi, n_test_sweep, seed):

    if dataset == "shifted_one_dim":
        (
            train,
            val,
            test,
            input_size,
            X_mean,
            X_std,
            y_mean,
            y_std,
        ), p_tests = get_shift_one_dim_dataloaders(
            dataset,
            n_train=n_train,
            p_train=p_train,
            p_test_lo=p_test_lo,
            p_test_hi=p_test_hi,
            n_test_sweep=n_test_sweep,
            seed=seed,
        )

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


    return train, val, test, input_size, X_mean, X_std, y_mean, y_std, p_tests
