import torch
import pdb
from pytorch_lightning.core.lightning import LightningModule
from loss import GenericLoss
import math
import numpy as np


class RockafellarUryasevModel(LightningModule):
    def __init__(self, input_size, loss, y_mean, y_scale):
        super().__init__()
        self.alpha_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        self.h_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        self.loss = loss
        self.sample_weights = None
        self.squared_loss = GenericLoss("squared_loss", y_mean=y_mean, y_scale=y_scale)

    def forward(self, x):
        h_out = self.h_net(x)
        alpha_out = self.alpha_net(x)
        return h_out, alpha_out

    def training_step(self, batch, batch_idx):
        x, y = batch
        h_out, alpha_out = self(x)
        l = self.loss(x, y, h_out, alpha_out).mean()
        tensorboard_logs = {
            "train_loss": l,
            "train_mse": self.squared_loss(h_out, y).mean(),
        }

        self.log_dict(tensorboard_logs, on_epoch=True)
        return {"loss": l, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        h_out, alpha_out = self(x)
        l = self.loss(x, y, h_out, alpha_out).mean()
        dic = {"val_loss": l, "val_mse": self.squared_loss(h_out, y).mean()}
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
        h_out, alpha_out = self(x)

        mse_loss = self.squared_loss(h_out, y, self.sample_weights)

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
                "test_loss": self.loss(
                    x, y, h_out, alpha_out, self.sample_weights
                ).mean(),
                "test_mse": mse_loss.sum(),
                "test_se": sums.std(),
            }
        else:
            dic = {
                "test_loss": self.loss(
                    x, y, h_out, alpha_out, self.sample_weights
                ).mean(),
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
