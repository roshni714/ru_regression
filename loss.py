import torch
import pdb


def squared_loss(y_hat, y_true):
    assert y_hat.shape == y_true.shape
    return (y_hat - y_true) ** 2


LOSS_DIC = {"squared_loss": squared_loss}


class RockafellarUryasevLoss:
    def __init__(self, loss, bound_function):
        self.loss = LOSS_DIC[loss]
        self.bound_function = bound_function

    def __call__(self, x, y, h_out, alpha_out, sample_weights=None):
        assert y.shape == h_out.shape
        assert y.shape == alpha_out.shape
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
    def __init__(self, loss, y_mean=None, y_scale=None):
        self.loss = LOSS_DIC[loss]
        self.y_mean = y_mean
        self.y_scale = y_scale

    def __call__(self, y_hat, y_true, sample_weights=None):
        assert y_hat.shape == y_true.shape
        if self.y_mean and self.y_scale:
            rescale_y_hat = self.y_scale * y_hat + self.y_mean
            rescale_y_true = self.y_scale * y_true + self.y_mean
            #            import pdb
            #            pdb.set_trace()
            l = self.loss(rescale_y_hat, rescale_y_true)
            if sample_weights is not None:
                #                import pdb
                #                pdb.set_trace()
                print("applying sample weights")
                assert l.shape == sample_weights.shape
                l *= sample_weights.to(l.get_device())

            return l
        else:
            l = self.loss(y_hat, y_true)
            if sample_weights is not None:
                #                import pdb
                #                pdb.set_trace()
                print("applying sample weights unrescaled")
                assert l.shape == sample_weights.shape
                l *= sample_weights.to(l.get_device())
            return l
