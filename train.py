from pytorch_lightning import Trainer, callbacks
from modules import BasicModel, RockafellarUryasevModel

from loss import RockafellarUryasevLoss, GenericLoss
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import os
import argh
from reporting import report_results
from utils import get_bound_function, get_dataset


def objective(
    dataset,
    p_train,
    p_test_lo,
    p_test_hi,
    n_test_sweep,
    method,
    loss,
    gamma,
    seed,
    epochs,
):
    train, val, tests, input_size, X_mean, X_std, y_mean, y_std, p_tests = get_dataset(
        dataset, p_train, p_test_lo, p_test_hi, n_test_sweep, seed
    )
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
            name="logs/{}_{}_{}_{}_p_train_{}_seed_{}".format(
                dataset, method, gamma, loss, p_train, seed
            ),
        )
        save_path = (
            "/scratch/users/rsahoo/models/{}_{}_{}_{}_p_train_{}_seed_{}.ckpt".format(
                dataset, method, int(gamma), loss, p_train, seed
            )
        )
    elif method == "erm":
        module = BasicModel
        loss_fn = GenericLoss(loss=loss)
        logger = TensorBoardLogger(
            save_dir="/scratch/users/rsahoo/runs",
            name="logs/{}_{}_{}_p_train_{}_seed_{}".format(
                dataset, method, loss, p_train, seed
            ),
        )
        save_path = (
            "/scratch/users/rsahoo/models/{}_{}_{}_p_train_{}_seed_{}.ckpt".format(
                dataset, method, loss, p_train, seed
            )
        )

    model = module(input_size=input_size, loss=loss_fn, y_mean=y_mean, y_scale=y_std)
    trainer = Trainer(
        gpus=1,
        #       checkpoint_callback=checkpoint_callback,
        max_epochs=epochs,
        logger=logger,
        val_check_interval=0.25,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_dataloader=train, val_dataloaders=val)
    res = []
    for i, test_loader in enumerate(tests):
        all_res = trainer.test(test_dataloaders=test_loader)
        all_res[0]["p_test"] = p_tests[i]
        res.append(all_res[0])
    trainer.save_checkpoint(save_path)
    return res


# Dataset
@argh.arg("--dataset", default="simulated")
@argh.arg("--p_train", default=0.2)
@argh.arg("--p_test_lo", default=0.1)
@argh.arg("--p_test_hi", default=0.8)
@argh.arg("--n_test_sweep", default=5)
# @argh.arg("--batch_size", default=128)
@argh.arg("--seed", default=0)
# Save
@argh.arg("--save", default="sim")

# Loss
@argh.arg("--method", default="ru_regression")
@argh.arg("--loss", default="squared_loss")
@argh.arg("--gamma", default=1.0)
# Epochs
@argh.arg("--epochs", default=40)
def main(
    dataset="simulated",
    p_train=0.2,
    p_test_lo=0.1,
    p_test_hi=0.8,
    n_test_sweep=5,
    seed=0,
    save="baseline_experiments",
    method="ru_regression",
    loss="squared_loss",
    gamma=1.0,
    epochs=40,
    #    batch_size=128,
):
    res = objective(
        dataset=dataset,
        p_train=p_train,
        p_test_lo=p_test_lo,
        p_test_hi=p_test_hi,
        n_test_sweep=n_test_sweep,
        seed=seed,
        method=method,
        loss=loss,
        gamma=gamma,
        epochs=epochs,
        #        batch_size=batch_size,
    )
    report_results(
        results=res,
        dataset=dataset,
        p_train=p_train,
        method=method,
        loss=loss,
        gamma=gamma,
        seed=seed,
        save=save,
    )


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
