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
        pred = torch.clamp(pred, min=min_depth)
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


class VPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "VP"

    def forward(self, pred, vp_loc, window_size=10, dmax=80):
        # premise: distance closer to a vanishing point is the greatest
        # conclusion: penalize for underpredicting distance in the region around a vanishing point
        # to make the penalty smoother, it gets weaker exponentially as pred moves away from vanishing point (window center)
        # QA: other funcs instead of exp to give a hint on how the distance should change? (ie, "closeness" to vp)
        window_size = (
            (window_size, window_size) if isinstance(window_size, int) else window_size
        )
        pred_window = self.slice_window(
            pred,
            img_size=pred.shape[-2:],
            window_center=vp_loc,
            window_size=window_size,
        )
        # loss = (pred_window - dmax) ** 2
        loss = torch.abs(pred_window - dmax).squeeze()
        idx_dist_matrix_within_window_from_its_center = (
            self.compute_idx_dist_inside_window(window_size=loss.shape[-2:])
        ).to(loss.device)

        loss *= idx_dist_matrix_within_window_from_its_center
        return loss.sum()

    def compute_idx_dist_inside_window(
        self, window_center_coord=None, window_size=(5, 5)
    ):
        # computed once per VP
        idx_dist_matrix_within_window_from_its_center = torch.zeros(window_size)
        window_center_coord = (
            window_center_coord
            if window_center_coord is not None
            else torch.tensor(window_size) // 2
        )
        if not isinstance(window_center_coord, torch.Tensor):
            window_center_coord = torch.tensor(window_center_coord)

        for i in range(window_size[0]):
            for j in range(window_size[1]):
                idx_dist = torch.linalg.norm(
                    window_center_coord - torch.tensor([i, j]).float()
                )
                # idx_dist_matrix_within_window_from_its_center[i, j] = torch.power(
                #     torch.e, -idx_dist
                # )
                idx_dist_matrix_within_window_from_its_center[i, j] = (
                    1 / (2 + idx_dist)
                )
                # idx_dist_matrix_within_window_from_its_center[i, j] = torch.exp(window_center_coord - torch.array([i, j]))
        return idx_dist_matrix_within_window_from_its_center

    def slice_window(self, img, img_size, window_center, window_size=(5, 5)):
        window_size = (
            (window_size, window_size) if isinstance(window_size, int) else window_size
        )
        h, w = img_size

        left_x = max(0, window_center[0] - window_size[0] // 2)
        if window_size[0] % 2 == 0:
            _right_x = window_center[0] + window_size[0] // 2
        else:
            _right_x = window_center[0] + window_size[0] // 2 + 1
        right_x = min(w, _right_x)

        top_y = max(0, window_center[1] - window_size[1] // 2)
        if window_size[1] % 2 == 0:
            _bottom_y = window_center[1] + window_size[1] // 2
        else:
            _bottom_y = window_center[1] + window_size[1] // 2 + 1
        bottom_y = min(h, _bottom_y)

        assert 3 in img.shape or 1 in img.shape, "Image must be either RGB or grayscale"
        if bottom_y > top_y:
            bottom_y, top_y = top_y, bottom_y

        bottom_y, top_y = int(bottom_y), int(top_y)
        left_x, right_x = int(left_x), int(right_x)

        if img.shape[-1] == 3:
            if len(img.shape) == 3:
                window = img[bottom_y:top_y, left_x:right_x, :]
            else:
                window = img[:, bottom_y:top_y, left_x:right_x, :]
        else:
            window = img[
                ...,
                bottom_y:top_y,
                left_x:right_x,
            ]
        return window
