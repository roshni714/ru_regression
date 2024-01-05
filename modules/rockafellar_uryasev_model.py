import torch
import pdb
from pytorch_lightning import LightningModule
from loss import GenericLoss
import math
import numpy as np


class RockafellarUryasevModel(LightningModule):
    def __init__(self, input_size, model_class, loss, y_mean, y_scale):
        super().__init__()
        if model_class == "neural_network":
            self.h_net = torch.nn.Sequential(
                torch.nn.Linear(input_size, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            )
        elif model_class == "linear":
            self.h_net = torch.nn.Sequential(
                torch.nn.Linear(input_size, 1),
            )

        #        import pdb
        #        pdb.set_trace()
        self.alpha_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        self.loss = loss
        self.mse = GenericLoss("squared_loss", y_scale=y_scale)

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        h_out = self.h_net(x)
        alpha_out = self.alpha_net(x)
        return h_out, alpha_out

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
            r = torch.ones(y.shape)  # / y.shape[0]
        elif len(batch) == 3:
            x, y, r = batch
        #            r /= r.sum()
        h_out, alpha_out = self(x)
        l = self.loss(x, y, h_out, alpha_out, r).mean()
        tensorboard_logs = {
            "train_loss": l,
        }

        self.log_dict(tensorboard_logs, on_epoch=True)
        return {"loss": l, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
            r = torch.ones(y.shape)  # / y.shape[0]
        elif len(batch) == 3:
            x, y, r = batch
        #            r /= r.sum()

        h_out, alpha_out = self(x)
        l = self.loss(x, y, h_out, alpha_out, r).mean()

        dic = {"val_loss": l}
        self.validation_step_outputs.append(dic)
        self.log("val_loss", l, prog_bar=True)
        return dic

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {}
        for key in outputs[0]:
            cal = torch.stack([x[key] for x in outputs]).mean()
            tensorboard_logs[key] = cal
        self.validation_step_outputs.clear()
        self.log_dict(tensorboard_logs)

        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        rng = np.random.RandomState(0)

        if len(batch) == 2:
            x, y = batch
            r = torch.ones(y.shape)  # /y.shape[0]
        elif len(batch) == 3:
            x, y, r = batch
        #            r /= r.sum()

        h_out, alpha_out = self(x)
        mse_sums = torch.zeros(5000)
        ru_loss_sums = torch.zeros(5000)
        inner_loss_sums = torch.zeros(5000)
        for i in range(5000):
            bootstrap_sample = rng.choice(len(y), size=len(y))
            if self.loss.loss_name == "poisson_nll":
                mse_loss = self.mse(
                    torch.exp(h_out)[bootstrap_sample, :],
                    y[bootstrap_sample, :],
                    r[bootstrap_sample, :],
                )
            else:
                mse_loss = self.mse(
                    h_out[bootstrap_sample, :],
                    y[bootstrap_sample, :],
                    r[bootstrap_sample, :],
                )

            inner_loss = self.loss.inner_loss(
                y[bootstrap_sample, :],
                h_out[bootstrap_sample, :],
                r[bootstrap_sample, :],
            )
            ru_loss = self.loss(
                x[bootstrap_sample, :],
                y[bootstrap_sample, :],
                h_out[bootstrap_sample, :],
                alpha_out[bootstrap_sample, :],
                r[bootstrap_sample, :],
            )

            sub_r = r[bootstrap_sample, :].sum().item()

            mse_sums[i] = mse_loss.mean().item()  # .sum().item() /sub_r
            ru_loss_sums[i] = ru_loss.mean().item()  # .sum().item() /sub_r
            inner_loss_sums[i] = inner_loss.mean().item()  # .sum().item() /sub_r

        inner_loss = self.loss.inner_loss(y, h_out, r)
        ru_loss = self.loss(x, y, h_out, alpha_out, r)
        if self.loss.loss_name == "poisson_nll":
            mse_loss = self.mse(torch.exp(h_out), y, r)
        else:
            mse_loss = self.mse(h_out, y, r)

        dic = {
            "test_ru_loss": ru_loss.mean(),
            "test_ru_loss_se": ru_loss_sums.std(),
            "test_loss": inner_loss.mean(),
            "test_loss_se": inner_loss_sums.std(),
            "test_mse": mse_loss.mean(),
            "test_mse_se": mse_sums.std(),
        }

        self.test_step_outputs.append(dic)
        return dic

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {}
        for key in outputs[0]:
            cal = torch.stack([x[key] for x in outputs]).mean()
            tensorboard_logs[key] = cal
        self.log_dict(tensorboard_logs)
        self.test_step_outputs.clear()
        return {
            "test_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
