import torch


def squared_loss(y_hat, y_true):
    return (y_hat - y_true) ** 2


LOSS_DIC = {"squared_loss": squared_loss}


class RockafellarUryasevLoss:
    def __init__(self, loss, bound_function):
        self.loss = LOSS_DIC[loss]
        self.bound_function = bound_function

    def __call__(self, x, y, h_out, alpha_out):

        bound_low = self.bound_function(x, low=True)
        bound_up = self.bound_function(x, up=True)

        loss_vals = self.loss(h_out, y)

        ru_loss = (
            bound_low * loss_vals
            + (1 - bound_low) * alpha_out
            + (bound_up - bound_low) * torch.nn.functional.relu(loss_vals - alpha_out)
        )
        return torch.mean(ru_loss)

class GenericLoss:

    def __init__(self, loss):
        self.loss = LOSS_DIC[loss]
