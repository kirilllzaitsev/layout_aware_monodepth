import numpy as np
import torch
from torchmetrics import Metric, MultioutputWrapper


class iRMSE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("value", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        self.value += torch.sqrt(torch.mean(((1.0 / target) - (1.0 / preds)) ** 2))

    def compute(self):
        return self.value


class iMAE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("value", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        self.value += torch.mean(torch.abs((1.0 / target) - (1.0 / preds)))

    def compute(self):
        return self.value


class TotalMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("mae", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("rmse", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("imae", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("irmse", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = (
            preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        )
        target = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
        self.mae += 0
        self.rmse += 0
        self.imae += 0
        self.irmse += 0
        self.total += target.shape[0]

    def compute(self):
        return self.mae, self.rmse, self.imae, self.irmse


def calc_metrics(gt, pred, mask=None, min_depth=1e-3):
    """Computes relevant metrics on non-zero pixels on ground truth depth map."""
    if mask is None:
        mask = gt > min_depth

    if isinstance(gt, torch.Tensor):
        gt = gt[mask].cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred[mask].detach().cpu().numpy()

    thresh = np.maximum((gt / (pred + 1e-8)), (pred / (gt + 1e-8)))
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25**2).mean()
    delta3 = (thresh < 1.25**3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(
        abs_rel=abs_rel,
        rmse=rmse,
        sq_rel=sq_rel,
        rmse_log=rmse_log,
        delta1=delta1,
        delta2=delta2,
        delta3=delta3,
        log_10=log_10,
        silog=silog,
    )


if __name__ == "__main__":
    metric = MultioutputWrapper(TotalMetric(), 4)
    preds = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    metric(preds, target)
    print(metric.compute())
    print(1)
