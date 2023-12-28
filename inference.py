import torch
from modules import BasicModel, RockafellarUryasevModel, JointRockafellarUryasevModel
from utils import get_bound_function, get_dataset
from reporting import report_regression
import argh
from utils import get_bound_function, get_dataset
from loss import RockafellarUryasevLoss, GenericLoss
import numpy as np


# Dataset
@argh.arg("--dataset", default="shifted_one_dim")
@argh.arg("--model_path", default="/scratch/users/rsahoo/models")
# @argh.arg("--batch_size", default=128)
@argh.arg("--seed", default=0)
# Save
@argh.arg("--save", default="sim")

# Loss
@argh.arg("--p_train", default=0.2)
@argh.arg("--method", default="ru_regression")
@argh.arg("--loss", default="squared_loss")
@argh.arg("--gamma", default=2.0)
def main(
    dataset="shifted_one_dim",
    model_path="/scratch/users/rsahoo/models",
    seed=0,
    save="inference_results",
    method="ru_regression",
    loss="squared_loss",
    p_train=0.2,
    gamma=2.0,
    epochs=40,
    #    batch_size=128,
):
    print("p_train", p_train)
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
        n_train=10000,
        d=1,
        unobserved=None,
        p_train=p_train,
        p_test_lo=0.2,
        p_test_hi=0.2,
        n_test_sweep=1,
        seed=seed,
    )

    if method == "ru_regression":
        module = RockafellarUryasevModel
        loss_fn = RockafellarUryasevLoss(
            loss=loss, bound_function=get_bound_function(gamma)
        )
        save_path = "{}/{}_{}_{}_{}_p_train_{}_seed_{}.ckpt".format(
            model_path, dataset, method, int(gamma), loss, p_train, seed
        )
    elif method == "joint_ru_regression":
        module = JointRockafellarUryasevModel
        loss_fn = RockafellarUryasevLoss(
            loss=loss, bound_function=get_bound_function(gamma)
        )
        save_path = "{}/{}_{}_{}_{}_p_train_{}_seed_{}.ckpt".format(
            model_path, dataset, method, int(gamma), loss, p_train, seed
        )

    elif method == "erm":
        module = BasicModel
        loss_fn = GenericLoss(loss=loss)
        save_path = "{}/{}_{}_{}_p_train_{}_seed_{}.ckpt".format(
            model_path, dataset, method, loss, p_train, seed
        )

    model = module.load_from_checkpoint(
        save_path, input_size=input_size, loss=loss_fn, y_mean=y_mean, y_scale=y_std
    )

    X = np.linspace(0.0, 10.0, 100)
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


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
