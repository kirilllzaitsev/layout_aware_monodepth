import torch
import torch.nn as nn


class MaskerMixin:
    def mask(self, pred, target, mask=None, interpolate=False, min_depth=1e-3):
        # if interpolate:
        #     pred = nn.functional.interpolate(
        #         pred, target.shape[-2:], mode="bilinear", align_corners=True
        #     )
        if mask is None:
            mask = target > min_depth

        pred = pred[mask]
        target = target[mask]
        return pred, target


class SILogLoss(nn.Module, MaskerMixin):
    """Following AdaBins"""

    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = "SILog"

    def forward(self, pred, target, mask=None, interpolate=False, min_depth=1e-3):
        pred, target = self.mask(pred, target, mask, interpolate, min_depth)
        g = torch.log(pred) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


class MSELoss(nn.Module, MaskerMixin):
    def __init__(self):
        super().__init__()
        self.name = "MSE"

    def forward(self, pred, target, mask=None, interpolate=False, min_depth=1e-3):
        pred, target = self.mask(pred, target, mask, interpolate, min_depth)
        return torch.mean((pred - target) ** 2)


class MAELoss(nn.Module, MaskerMixin):
    def __init__(self):
        super().__init__()
        self.name = "MAE"

    def forward(self, pred, target, mask=None, interpolate=False, min_depth=1e-3):
        pred, target = self.mask(pred, target, mask, interpolate, min_depth)
        return torch.mean(torch.abs(pred - target))
