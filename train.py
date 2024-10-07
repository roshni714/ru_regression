from pytorch_lightning import Trainer, callbacks
from modules import RockafellarUryasevModel, JointRockafellarUryasevModel

from loss import RockafellarUryasevLoss, GenericLoss
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import os
import argh
from reporting import report_results, report_regression
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
    use_train_weights,
    n_train,
    n_test_sweep,
    method,
    model_class,
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
        p_train,
        p_test_lo,
        p_test_hi,
        n_test_sweep,
        use_train_weights,
        loss,
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
        save_path = "{}/{}_{}_{}_{}_p_train_{}_seed_{}.ckpt".format(
            model_path, dataset, method, int(gamma), loss, p_train, seed
        )
    elif method == "joint_ru_regression":
        module = JointRockafellarUryasevModel
        loss_fn = RockafellarUryasevLoss(
            loss=loss, bound_function=get_bound_function(gamma)
        )
        logger = TensorBoardLogger(
            save_dir=run_path,
            name="logs/{}_{}_{}_{}_p_train_{}_seed_{}".format(
                dataset, method, gamma, loss, p_train, seed
            ),
        )
        save_path = "{}/{}_{}_{}_{}_p_train_{}_seed_{}.ckpt".format(
            model_path, dataset, method, int(gamma), loss, p_train, seed
        )

    torch.manual_seed(0)

    if "mimic" in dataset:
        normalize = True
    else:
        normalize = False
    model = module(
        input_size=input_size,
        model_class=model_class,
        loss=loss_fn,
        y_mean=y_mean,
        y_scale=y_std,
        normalize=normalize
    )

    trainer = Trainer(
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        max_epochs=epochs,
        logger=logger,
        val_check_interval=0.25,
    )

    torch.manual_seed(0)
    trainer.fit(model, train_dataloaders=train, val_dataloaders=val)
    trainer.save_checkpoint(save_path)
    res = []
    trainer

    if "mimic" in dataset:
        target_res = trainer.test(dataloaders=tests[0], ckpt_path="best")
        val_res = trainer.test(dataloaders=tests[1], ckpt_path="best")
        target_res[0]["heldout_train_ru_loss"] = val_res[0]["test_ru_loss"]
        target_res[0]["heldout_train_ru_loss_se"] = val_res[0]["test_ru_loss_se"]
        target_res[0]["heldout_train_mse"] = val_res[0]["test_mse"]
        target_res[0]["heldout_train_mse_se"] = val_res[0]["test_mse_se"]
        target_res[0]["heldout_train_loss"] = val_res[0]["test_loss"]
        target_res[0]["heldout_train_loss_se"] = val_res[0]["test_loss_se"]

        res.append(target_res[0])

    elif "survey" in dataset:
        target_res = trainer.test(dataloaders=tests[0], ckpt_path="best")
        val_res = trainer.test(dataloaders=tests[1], ckpt_path="best")
        target_res[0]["hps_ru_loss"] = val_res[0]["test_ru_loss"]
        target_res[0]["hps_ru_loss_se"] = val_res[0]["test_ru_loss_se"]
        target_res[0]["hps_mse"] = val_res[0]["test_mse"]
        target_res[0]["hps_mse_se"] = val_res[0]["test_mse_se"]
        target_res[0]["hps_loss"] = val_res[0]["test_loss"]
        target_res[0]["hps_loss_se"] = val_res[0]["test_loss_se"]

        res.append(target_res[0])
    else:
        for i, test_loader in enumerate(tests):
            all_res = trainer.test(dataloaders=test_loader, ckpt_path="best")
            all_res[0]["p_test"] = p_tests[i]
            res.append(all_res[0])
    return model, res, X_mean, X_std, y_mean, y_std


def inference(
    model, method, loss, p_train, dataset, gamma, X_mean, X_std, y_mean, y_std, seed
):
    save = "inference_results"
    if dataset == "heteroscedastic_one_dim":
        max_X = 10.0
    else:
        max_X = 6.0

    X = np.linspace(0.0, max_X, 100)
    r_X = (X - X_mean) / X_std
    r_X = torch.Tensor(r_X.reshape(-1, 1))

    if method == "ru_regression":
        r_y, alpha = model(r_X)
        r_y = r_y.detach().numpy()
    elif method == "joint_ru_regression":
        r_y = model(r_X)
        r_y = r_y.detach().numpy()
        alpha = model.alpha.detach().numpy().item() * np.ones(r_y.shape)
    else:
        r_y = model(r_X).detach().numpy()
        alpha = np.zeros(r_y.shape)
    y = r_y * y_std + y_mean
    alpha *= y_std**2
    report_regression(
        dataset=dataset,
        seed=seed,
        save=save,
        method=method,
        loss=loss,
        p_train=p_train,
        gamma=gamma,
        X=X,
        y=y,
        alpha=alpha,
    )


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
@argh.arg("--use_train_weights")
# @argh.arg("--batch_size", default=128)
@argh.arg("--seed", default=0)
# Save
@argh.arg("--save", default="sim")

# Loss
@argh.arg("--method", default="ru_regression")
@argh.arg("--loss", default="squared_loss")
@argh.arg("--gamma", default=1.0)
@argh.arg("--model_class", default="neural_network")
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
    n_train=10000,
    n_test_sweep=5,
    use_train_weights=False,
    seed=0,
    save="baseline_experiments",
    method="ru_regression",
    loss="squared_loss",
    model_class="neural_network",
    gamma=1.0,
    epochs=40,
    #    batch_size=128,
):
    model, res, X_mean, X_std, y_mean, y_std = objective(
        dataset=dataset,
        run_path=run_path,
        model_path=model_path,
        d=d,
        p_train=p_train,
        p_test_lo=p_test_lo,
        p_test_hi=p_test_hi,
        use_train_weights=use_train_weights,
        n_train=n_train,
        n_test_sweep=n_test_sweep,
        model_class=model_class,
        seed=seed,
        method=method,
        loss=loss,
        gamma=gamma,
        epochs=epochs,
        #        batch_size=batch_size,
    )

    if "one_dim" in dataset:
        inference(
            model=model,
            method=method,
            loss=loss,
            p_train=p_train,
            dataset=dataset,
            gamma=gamma,
            X_mean=X_mean,
            X_std=X_std,
            y_mean=y_mean,
            y_std=y_std,
            seed=seed,
        )
    report_results(
        results=res,
        dataset=dataset,
        n_train=n_train,
        p_train=p_train,
        d=d,
        method=method,
        loss=loss,
        model_class=model_class,
        use_train_weights=use_train_weights,
        gamma=gamma,
        seed=seed,
        save=save,
    )


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
