from pytorch_lightning import Trainer, callbacks
from modules import BasicModel, RockafellarUryasevModel
from data_loaders import (
    get_shift_dataloaders,
)
from loss import RockafellarUryasevLoss
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import os
import argh
from reporting import report_results


def get_dataset(dataset, seed, batch_size):
    train, val, test, input_size = get_shift_dataloaders(
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


def objective(dataset, loss, gamma, seed, epochs, batch_size):
    train, val, test, input_size = get_dataset(dataset, seed, batch_size)
    checkpoint_callback = callbacks.model_checkpoint.ModelCheckpoint(
        "/scratch/rsahoo/models/{}_{}_seed_{}/".format(dataset, loss, seed),
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    logger = TensorBoardLogger(
        save_dir="/scratch/rsahoo/runs",
        name="logs/{}_{}_seed_{}".format(dataset, loss, seed),
    )

    if method == "ru_regression":
        module= RockafellarUryasevModel
        loss_fn = RockafellarUryasevLoss(
        loss=loss, bound_function=get_bound_function(gamma)
        )
    elif method == "erm":
        module = BasicModel
        loss_fn = GenericLoss(loss=loss)
    model = module(input_size=input_size, loss=loss_fn)
    trainer = Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        max_epochs=epochs,
        logger=logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_dataloader=train, val_dataloaders=val)
    trainer.test(test_dataloaders=test)

    return model


# Dataset
@argh.arg("--dataset", default="simulated")
@argh.arg("--batch_size", default=128)
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
    loss="squared_loss",
    gamma=2.0,
    epochs=40,
    batch_size=128,
):
    model = objective(
        dataset=dataset,
        seed=seed,
        method=method,
        loss=loss,
        gamma=gamma,
        epochs=epochs,
        batch_size=batch_size,
    )
    report_results(model, dataset, loss, gamma, seed, save)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
