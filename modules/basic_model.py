import torch
import pdb
from pytorch_lightning.core.lightning import LightningModule
from loss import GenericLoss
import math
import numpy as np


class BasicModel(LightningModule):
    def __init__(self, input_size, loss, y_mean, y_scale):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        self.loss = loss
        self.mse = GenericLoss("squared_loss", y_mean, y_scale)
        self.sample_weights = None

    def forward(self, x):
        y_hat = self.net(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        l = self.loss(y_hat, y).mean()
        tensorboard_logs = {"train_loss": l, "train_mse": self.mse(y_hat, y).mean()}
        dic = {"loss": l, "log": tensorboard_logs}
        self.log_dict(tensorboard_logs, on_epoch=True)

        return dic

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # import pdb
        # pdb.set_trace()
        l = self.loss(y_hat, y).mean()
        dic = {"val_loss": l, "val_mse": self.mse(y_hat, y).mean()}
        return dic

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {}
        for key in outputs[0]:
            cal = torch.stack([x[key] for x in outputs]).mean()
            tensorboard_logs[key] = cal
        self.log_dict(tensorboard_logs)

        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse_loss = self.mse(y_hat, y, self.sample_weights)

        if self.sample_weights is not None:
            rng = np.random.RandomState(0)
            mse_loss_np = mse_loss.detach().cpu().numpy().flatten()
            sums = torch.zeros(5000)
            for i in range(5000):
                bootstrap_sample = rng.choice(mse_loss_np, size=mse_loss.shape[0])
                sums[i] = bootstrap_sample.sum().item()
            #        import pdb
            #        pdb.set_trace()
            #        import pdb
            #        pdb.set_trace()
            dic = {
                "test_loss": self.loss(y_hat, y, self.sample_weights).mean(),
                "test_mse": mse_loss.sum(),
                "test_se": sums.std(),
            }
        else:
            dic = {
                "test_loss": self.loss(y_hat, y, self.sample_weights).mean(),
                "test_mse": mse_loss.mean(),
            }
        return dic

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {}
        for key in outputs[0]:
            cal = torch.stack([x[key] for x in outputs]).mean()
            tensorboard_logs[key] = cal
        self.log_dict(tensorboard_logs)
        return {
            "test_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
