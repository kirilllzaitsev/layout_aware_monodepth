import torch
from kbnet import eval_utils
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
        self.mae += eval_utils.mean_abs_err(1000.0 * preds, 1000.0 * target)
        self.rmse += eval_utils.root_mean_sq_err(1000.0 * preds, 1000.0 * target)
        self.imae += eval_utils.inv_mean_abs_err(0.001 * preds, 0.001 * target)
        self.irmse += eval_utils.inv_root_mean_sq_err(0.001 * preds, 0.001 * target)
        self.total += target.shape[0]

    def compute(self):
        return self.mae, self.rmse, self.imae, self.irmse


if __name__ == "__main__":
    metric = MultioutputWrapper(TotalMetric(), 4)
    preds = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    metric(preds, target)
    print(metric.compute())
    print(1)
