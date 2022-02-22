import torch
import pdb
from pytorch_lightning.core.lightning import LightningModule
from loss import GenericLoss


class BasicModel(LightningModule):
    def __init__(self, input_size, loss):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
        )
        self.loss = loss
        self.squared_loss = GenericLoss("squared_loss")

    def forward(self, x):
        y_hat = self.net(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        l = self.loss(y_hat, y)
        tensorboard_logs = {"train_loss": l, "train_mse": self.squared_loss(y_hat, y)}
        dic = {"loss": l, "log": tensorboard_logs}
        self.log_dict(tensorboard_logs, on_epoch=True)

        return dic

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        l = self.loss(y_hat, y)
        dic = {"val_loss": l, "val_mse": self.squared_loss(y_hat, y)}
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
        dic = {
            "test_loss": self.loss(y_hat, y),
            "test_mse": self.squared_loss(y_hat, y),
        }
        return dic

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {}
        for key in outputs[0]:
            cal = torch.stack([x[key] for x in outputs]).mean()
            tensorboard_logs[key] = cal
            setattr(self, key, float(cal))
        self.log_dict(tensorboard_logs)
        return {
            "test_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
