import torch
from modules import BasicModel, RockafellarUryasevModel
from data_loaders import (
    get_shift_dataloaders,
    get_shift_oracle_dataloaders,
)
from reporting import report_regression
import argh
from utils import get_bound_function, get_dataset
from loss import RockafellarUryasevLoss, GenericLoss
import numpy as np

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

    _, _, _, input_size, X_mean, X_std, y_mean, y_std = get_dataset(dataset, seed)

    if method == "ru_regression":
        module = RockafellarUryasevModel
        loss_fn = RockafellarUryasevLoss(
            loss=loss, bound_function=get_bound_function(gamma)
        )
        save_path = "/scratch/users/rsahoo/models/{}_{}_{}_{}_seed_{}.ckpt".format(
            dataset, method, int(gamma), loss, seed
        )

    elif method == "erm":
        module = BasicModel
        loss_fn = GenericLoss(loss=loss)
        save_path = "/scratch/users/rsahoo/models/{}_{}_{}_seed_{}.ckpt".format(
            dataset, method, loss, seed
        )

    model = module.load_from_checkpoint(save_path, input_size=input_size, loss=loss_fn)

    X = np.linspace(0.0, 10.0, 100)
    r_X = (X - X_mean) / X_std
    r_X = torch.Tensor(r_X.reshape(-1, 1))
    print(X)
    if method=="ru_regression":
        r_y, _= model(r_X)
        r_y = r_y.detach().numpy()
    else:
        r_y = model(r_X).detach().numpy()
    print(r_y)
    y = r_y * y_std + y_mean
    print(y)
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
