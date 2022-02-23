from data_loaders import (
    get_shift_dataloaders,
    get_shift_oracle_dataloaders,
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


def get_dataset(dataset, seed):

    if dataset == "shifted":
        (
            train,
            val,
            test,
            input_size,
            X_mean,
            X_std,
            y_mean,
            y_std,
        ) = get_shift_dataloaders(dataset, seed=seed)
    elif dataset == "shifted_oracle":
        (
            train,
            val,
            test,
            input_size,
            X_mean,
            X_std,
            y_mean,
            y_std,
        ) = get_shift_oracle_dataloaders(dataset, seed=seed)

    return train, val, test, input_size, X_mean, X_std, y_mean, y_std
