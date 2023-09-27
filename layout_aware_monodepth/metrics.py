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


def calc_metrics(gt, pred, mask=None, min_depth=1e-3, depth_magnitude_factor=None):
    """Computes relevant metrics on non-zero pixels on ground truth depth map."""
    if depth_magnitude_factor is not None:
        gt = gt * depth_magnitude_factor
        pred = pred * depth_magnitude_factor

    if mask is None:
        mask = gt > min_depth

    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    gt = gt[mask]
    pred = pred[mask]

    thresh = np.maximum((gt / (pred + 1e-8)), (pred / (gt + 1e-8)))
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25**2).mean()
    delta3 = (thresh < 1.25**3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    mae = np.mean(np.abs(gt - pred))
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
        mae=mae,
        sq_rel=sq_rel,
        rmse_log=rmse_log,
        delta1=delta1,
        delta2=delta2,
        delta3=delta3,
        log_10=log_10,
        silog=silog,
    )


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {
            f"{key}": round(value.get_value(), 4) for key, value in self._dict.items()
        }

    def __str__(self):
        return str(self.get_value())


if __name__ == "__main__":
    metric = MultioutputWrapper(TotalMetric(), 4)
    preds = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    metric(preds, target)
    print(metric.compute())
    print(1)
