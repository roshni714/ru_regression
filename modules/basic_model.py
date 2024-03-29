import torch
import pdb
from pytorch_lightning import LightningModule
from loss import GenericLoss
import math
import numpy as np

torch.manual_seed(0)


class BasicModel(LightningModule):
    def __init__(self, input_size, model_class, loss, y_mean, y_scale):
        super().__init__()
        if model_class == "neural_network":
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_size, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            )
        else:
            self.net = torch.nn.Sequential(torch.nn.Linear(input_size, 1))

        self.loss = loss
        self.mse = GenericLoss("squared_loss", y_scale)
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        y_hat = self.net(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
            r = torch.ones(y.shape) / y.shape[0]
        elif len(batch) == 3:
            x, y, r = batch
            r /= r.sum()

        y_hat = self(x)
        l = self.loss(y_hat, y, r).sum()
        tensorboard_logs = {"train_loss": l}
        dic = {"loss": l, "log": tensorboard_logs}
        self.log_dict(tensorboard_logs, on_epoch=True)
        return dic

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
            r = torch.ones(y.shape) / y.shape[0]
        elif len(batch) == 3:
            x, y, r = batch
            r /= r.sum()

        y_hat = self(x)
        # import pdb
        # pdb.set_trace()
        l = self.loss(y_hat, y, r).sum()
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
        self.log_dict(tensorboard_logs)
        self.validation_step_outputs.clear()
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
            r = torch.ones(y.shape) / y.shape[0]
        elif len(batch) == 3:
            x, y, r = batch
            r /= r.sum()

        y_hat = self(x)
        if self.loss.name == "poisson_nll":
            mse_loss = self.mse(torch.exp(y_hat), y, r)
        else:
            mse_loss = self.mse(y_hat, y, r)
        rng = np.random.RandomState(0)
        mse_loss_np = mse_loss.detach().cpu().numpy().flatten()
        sums = torch.zeros(5000)
        for i in range(5000):
            bootstrap_sample = rng.choice(
                list(range(len(mse_loss))), size=mse_loss.shape[0]
            )
            sub_mse = mse_loss[bootstrap_sample]
            sub_r = r[bootstrap_sample]
            sums[i] = sub_mse.sum().item() / sub_r.sum().item()

        dic = {
            "test_loss": self.loss(y_hat, y, r).sum(),
            "test_mse": mse_loss.sum(),
            "test_se": sums.std(),
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
