import torch
from modules import BasicModel, RockafellarUryasevModel
from data_loaders import (
    get_shift_dataloaders,
    get_shift_oracle_dataloaders,
)
from reporting import report_regression
import argh


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


# Dataset
@argh.arg("--dataset", default="simulated")
# @argh.arg("--batch_size", default=128)
@argh.arg("--seed", default=0)
# Save
@argh.arg("--save", default="sim")

# Loss
@argh.arg("--method", default="ru_regression")
@argh.arg("--loss", default="squared_loss")
@argh.arg("--gamma", default=2.0)
def main(
    dataset="simulated",
    seed=0,
    save="baseline_experiments",
    method="ru_regression",
    loss="squared_loss",
    gamma=2.0,
    epochs=40,
    #    batch_size=128,
):

    _, _, _, _, X_mean, X_std, y_mean, y_std = get_dataset(dataset, seed)

    if method == "ru_regression":
        module = RockafellarUryasevModel
        save_path = "/scratch/users/rsahoo/models/{}_{}_{}_{}_seed_{}.ckpt".format(
            dataset, method, int(gamma), loss, seed
        )

    elif method == "erm":
        module = BasicModel
        save_path = "/scratch/users/rsahoo/models/{}_{}_{}_seed_{}.ckpt".format(
            dataset, method, loss, seed
        )

    model = module.load_from_checkpoint(save_path)

    X = np.linspace(0.0, 10.0, 100)
    r_X = (X - X_mean) / X_std
    r_X = torch.Tensor(r_X.reshape(-1, 1))

    r_y = model(r_X)
    y = r_y * y_std + y_mean

    report_regression(
        dataset=dataset,
        seed=seed,
        save=save,
        method=method,
        loss=loss,
        gamma=gamma,
        X=X,
        y=y,
    )


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
