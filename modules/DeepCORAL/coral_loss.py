import torch
import torch.nn as nn
from modules.utils import normalize_tensor


class DeepCORAL(nn.Module):
    # deep coral with optional EMA for the covariance matrices
    def __init__(self, ema=1.0, detach_prev=False, *args, **kwargs):
        """
        :param ema: when 1.0, will not use EMA (use only current batch)
            otherwise, cov_crt = cov_new * self.ema + cov_prev * (1.0 - self.ema)
        :param detach_prev:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.Cs = None
        self.Ct = None
        self.ema = max(0.0, min(1.0, ema))
        self.detach_prev = detach_prev

    def forward(self, xs, xt):
        if self.ema is None or self.ema == 1.0:
            # no need to apply EMA
            return coral_loss(xs, xt)

        # compute the current cov matrices
        self.Cs = self.update_cov(self.Cs, xs)
        self.Ct = self.update_cov(self.Ct, xt)

        # compute CORAL loss
        d = xs.shape[0]
        return coral_loss_cov(self.Cs, self.Ct, d)

    def update_cov(self, cov_prev, x):
        cov_new = compute_covariances(x)
        if cov_prev is None:
            return cov_new
        cov_prev = cov_prev.detach()  # must detach otherwise backprop twice
        return cov_new * self.ema + cov_prev * (1.0 - self.ema)


def compute_covariances(x):
    # source covariance
    x = normalize_tensor(x)
    cov = torch.mean(x, 0, keepdim=True) - x
    cov = cov.t() @ cov
    return cov


def coral_loss_cov(Cs, Ct, d):
    # Frobenius norm = sqrt of the sum of all elements squared in the cov difference matrix
    # similar to L2 distance
    # https://mathworld.wolfram.com/FrobeniusNorm.html
    loss = torch.norm(Cs - Ct, p='fro')

    # original DeepCoral uses "fro/(4*d*d)", but we use fro/d which seems to be more consistent across varying d
    # this change requires coral loss weight to be decreased.
    # ex: for d=1000, when using fro/4dd, lambda=~1000 to compensate for dividing by 4dd
    # when d=1000, when using fro/d, lambda=1 is adequate

    return loss / d


# "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
# https://arxiv.org/pdf/1607.01719.pdf
def coral_loss(xs, xt):
    """
    Computes coral loss between b x d tensors (batch_size x feature_dim)
    :param xs:
    :param xd:
    :return:
    """
    Cs = compute_covariances(xs)
    Ct = compute_covariances(xt)

    d = xs.shape[1]  # feature dim
    return coral_loss_cov(Cs, Ct, d)
