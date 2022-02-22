from pytorch_lightning import Trainer, callbacks
from modules import BasicModel, RockafellarUryasevModel
from data_loaders import (
    get_shift_dataloaders,
    get_shift_oracle_dataloaders,
)
from loss import RockafellarUryasevLoss, GenericLoss
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import os
import argh
from reporting import report_results


def get_dataset(dataset, seed):

    if dataset == "shifted":
        train, val, test, input_size, _, _, _, _ = get_shift_dataloaders(
            dataset, seed=seed
        )
    elif dataset == "shifted_oracle":
        train, val, test, input_size, _, _, _, _ = get_shift_oracle_dataloaders(
            dataset, seed=seed
        )

    return train, val, test, input_size


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


def objective(dataset, method, loss, gamma, seed, epochs):
    train, val, test, input_size = get_dataset(dataset, seed)
    #    checkpoint_callback = callbacks.model_checkpoint.ModelCheckpoint(
    #        ),
    #        monitor="val_loss",
    #        save_top_k=1,
    #        mode="min",
    #    )

    if method == "ru_regression":
        module = RockafellarUryasevModel
        loss_fn = RockafellarUryasevLoss(
            loss=loss, bound_function=get_bound_function(gamma)
        )
        logger = TensorBoardLogger(
            save_dir="/scratch/users/rsahoo/runs",
            name="logs/{}_{}_{}_{}_seed_{}".format(dataset, method, gamma, loss, seed),
        )
        save_path = "/scratch/users/rsahoo/models/{}_{}_{}_{}_seed_{}.ckpt".format(
            dataset, method, int(gamma), loss, seed
        )
    elif method == "erm":
        module = BasicModel
        loss_fn = GenericLoss(loss=loss)
        logger = TensorBoardLogger(
            save_dir="/scratch/users/rsahoo/runs",
            name="logs/{}_{}_{}_seed_{}".format(dataset, method, loss, seed),
        )
        save_path = "/scratch/users/rsahoo/models/{}_{}_{}_seed_{}.ckpt".format(
            dataset, method, loss, seed
        )

    model = module(input_size=input_size, loss=loss_fn)
    trainer = Trainer(
        gpus=1,
        #       checkpoint_callback=checkpoint_callback,
        max_epochs=epochs,
        logger=logger,
        val_check_interval=0.25,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_dataloader=train, val_dataloaders=val)
    trainer.test(test_dataloaders=test)
    trainer.save_checkpoint(save_path)
    return model


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
# Epochs
@argh.arg("--epochs", default=40)
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
    model = objective(
        dataset=dataset,
        seed=seed,
        method=method,
        loss=loss,
        gamma=gamma,
        epochs=epochs,
        #        batch_size=batch_size,
    )
    report_results(model, dataset, method, loss, gamma, seed, save)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
