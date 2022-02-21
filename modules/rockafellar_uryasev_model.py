import torch
import pdb
from pytorch_lightning.core.lightning import LightningModule


class RockafellarUryasevModel(LightningModule):
    def __init__(self, input_size, loss):
        super().__init__()
        self.alpha_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
        )
        self.h_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
        )
        self.loss = loss

    def forward(self, x):
        h_out = self.h_net(x)
        alpha_out = self.alpha_net(x)
        return h_out, alpha_out

    def training_step(self, batch, batch_idx):
        x, y = batch
        h_out, alpha_out = self(x)
        l = self.loss(x, y, h_out, alpha_out)
        tensorboard_logs = {"train_loss": l}
        return {"loss": l, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        h_out, alpha_out = self(x)
        l = self.loss(x, y, h_out, alpha_out)
        return {"val_loss": l}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {}
        for key in outputs[0]:
            cal = torch.stack([x[key] for x in outputs]).mean()
            tensorboard_logs[key] = cal
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        h_out, alpha_out = self(x)
        dic = {
            "test_loss": self.loss(x, y, h_out, alpha_out),
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
