import torch
import pdb


def squared_loss(y_hat, y_true):
    assert y_hat.shape == y_true.shape
    return (y_hat - y_true) ** 2


def poisson_nll(y_hat, y_true):
    return -y_hat * y_true + torch.exp(y_hat)


LOSS_DIC = {"squared_loss": squared_loss, "poisson_nll": poisson_nll}


class RockafellarUryasevLoss:
    def __init__(self, loss, bound_function):
        self.loss = LOSS_DIC[loss]
        self.bound_function = bound_function
        self.loss_name = loss

    def inner_loss(self, y, h_out, sample_weights=None):
        if sample_weights is not None:
            return self.loss(h_out, y) * sample_weights.to(h_out.get_device())
        else:
            return self.loss(h_out, y)

    def __call__(self, x, y, h_out, alpha_out, sample_weights=None):
        assert y.shape == h_out.shape
        assert y.shape == alpha_out.shape or alpha_out.shape[0] == 1
        bound_low = self.bound_function(x, low=True)
        bound_up = self.bound_function(x, up=True)

        loss_vals = self.loss(h_out, y)

        ru_loss = (
            bound_low * loss_vals
            + (1 - bound_low) * alpha_out
            + (bound_up - bound_low) * torch.nn.functional.relu(loss_vals - alpha_out)
        )
        if sample_weights is not None:
            assert ru_loss.shape == sample_weights.shape
            ru_loss *= sample_weights.to(ru_loss.get_device())

        return ru_loss


class GenericLoss:
    def __init__(self, loss, y_scale=None):
        self.loss = LOSS_DIC[loss]
        self.y_scale = y_scale
        self.name = loss

    def __call__(self, y_hat, y_true, sample_weights=None):
        assert y_hat.shape == y_true.shape
        if self.y_scale is not None:
            rescale_y_hat = self.y_scale * y_hat
            rescale_y_true = self.y_scale * y_true
            l = self.loss(rescale_y_hat, rescale_y_true)
        else:
            l = self.loss(y_hat, y_true)
        if sample_weights is not None:
            assert l.shape == sample_weights.shape
            l *= sample_weights.to(l.get_device())
        return l
