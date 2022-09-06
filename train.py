from pytorch_lightning import Trainer, callbacks
from modules import BasicModel, RockafellarUryasevModel

from loss import RockafellarUryasevLoss, GenericLoss
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import os
import argh
from reporting import report_results
from utils import get_bound_function, get_dataset
import torch

torch.manual_seed(0)

def objective(
    dataset,
    run_path,
    model_path,
    d,
    p_train,
    p_test_lo,
    p_test_hi,
    unobserved,
    n_train,
    n_test_sweep,
    method,
    loss,
    gamma,
    seed,
    epochs,
):
    (
        train,
        val,
        tests,
        input_size,
        X_mean,
        X_std,
        y_mean,
        y_std,
        p_tests,
        test_weights,
    ) = get_dataset(
        dataset,
        n_train,
        d,
        unobserved,
        p_train,
        p_test_lo,
        p_test_hi,
        n_test_sweep,
        seed,
    )

    checkpoint_callback = callbacks.model_checkpoint.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    if method == "ru_regression":
        module = RockafellarUryasevModel
        loss_fn = RockafellarUryasevLoss(
            loss=loss, bound_function=get_bound_function(gamma)
        )
        logger = TensorBoardLogger(
            save_dir=run_path,
            name="logs/{}_{}_{}_{}_p_train_{}_seed_{}".format(
                dataset, method, gamma, loss, p_train, seed
            ),
        )
        save_path = (
            "{}/{}_{}_{}_{}_p_train_{}_seed_{}.ckpt".format(model_path,
                dataset, method, int(gamma), loss, p_train, seed
            )
        )
    elif method == "erm":
        module = BasicModel
        loss_fn = GenericLoss(loss=loss)
        logger = TensorBoardLogger(
            save_dir=run_path,
            name="logs/{}_{}_{}_p_train_{}_seed_{}".format(
                dataset, method, loss, p_train, seed
            ),
        )
        save_path = (
            "{}/{}_{}_{}_p_train_{}_seed_{}.ckpt".format(model_path,
                dataset, method, loss, p_train, seed
            )
        )

    model = module(input_size=input_size, loss=loss_fn, y_mean=y_mean, y_scale=y_std)
    trainer = Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        max_epochs=epochs,
        logger=logger,
        val_check_interval=0.1,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dataloader=train, val_dataloaders=val)
    trainer.save_checkpoint(save_path)
    res = []
    trainer

    if dataset == "mimic":
        for i, p_test in enumerate(p_tests):
            model.sample_weights = test_weights[i]
            all_res = trainer.test(test_dataloaders=tests[0], ckpt_path="best")
            all_res[0]["p_test"] = p_test
            all_res[0]["unobserved"] = unobserved
            res.append(all_res[0])
    else:
        for i, test_loader in enumerate(tests):
            all_res = trainer.test(test_dataloaders=test_loader, ckpt_path="best")
            all_res[0]["p_test"] = p_tests[i]
            res.append(all_res[0])
    return res


# Dataset
@argh.arg("--dataset", default="simulated")
@argh.arg("--model_path", default="/scratch/users/rsahoo/models")
@argh.arg("--run_path", default="/scratch/users/rsahoo/runs") 
@argh.arg("--d", default=2)
@argh.arg("--p_train", default=0.2)
@argh.arg("--p_test_lo", default=0.1)
@argh.arg("--p_test_hi", default=0.8)
@argh.arg("--n_train", default=7000)
@argh.arg("--n_test_sweep", default=5)
@argh.arg("--unobserved", default=None)
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
    dataset="shifted_one_dim",
    run_path="/scratch/users/rsahoo/runs",
    model_path="/scratch/users/rsahoo/models",
    d=2,
    p_train=0.2,
    p_test_lo=0.1,
    p_test_hi=0.8,
    unobserved=None,
    n_train=7000,
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
        run_path=run_path,
        model_path=model_path,
        d=d,
        p_train=p_train,
        p_test_lo=p_test_lo,
        p_test_hi=p_test_hi,
        unobserved=unobserved,
        n_train=n_train,
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
        n_train=n_train,
        p_train=p_train,
        d=d,
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
