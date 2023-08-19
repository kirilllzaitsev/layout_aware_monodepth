import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    """Following AdaBins"""

    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = "SILog"

    def forward(self, pred, target, mask=None, interpolate=False, min_depth=1e-3):
        if interpolate:
            pred = nn.functional.interpolate(
                pred, target.shape[-2:], mode="bilinear", align_corners=True
            )

        if mask is None:
            mask = target > min_depth

        pred = pred[mask]
        target = target[mask]
        g = torch.log(pred) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None, interpolate=False, min_depth=1e-3):
        if interpolate:
            pred = nn.functional.interpolate(
                pred, target.shape[-2:], mode="bilinear", align_corners=True
            )

        if mask is None:
            mask = target > min_depth

        pred = pred[mask]
        target = target[mask]
        return torch.mean((pred - target) ** 2)
