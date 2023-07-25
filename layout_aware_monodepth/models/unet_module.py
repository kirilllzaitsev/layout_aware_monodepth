from typing import Any

import numpy as np
import torch
from lightning import LightningModule

from layout_aware_monodepth.metrics import TotalMetric
from layout_aware_monodepth.model import DepthModel


class UnetLitModule(LightningModule):
    def __init__(
        self,
        depth_net: DepthModel,
        transforms: dict,
        config: Any,
        ckpt_path: str = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.depth_net = depth_net
        self.config = config
        self.train_transforms = transforms["train"]
        self.val_transforms = transforms["val"]

        self.exp = self.logger.experiment

        self.train_total = TotalMetric()
        self.val_total = TotalMetric()
        self.test_total = TotalMetric()
        self.do_log_with_logger = self.logger is not None
        self.to(self.device)

    def load_from_checkpoint(self, ckpt_path):
        self.depth_net.restore_model(ckpt_path)
        return self

    def to(self, device):
        self.depth_net.to(device)

    def forward(self, x: torch.Tensor):
        return torch.randn(x.shape[0], 10)

    def eval_model_step(self, batch: Any, transforms):
        output_depth = None
        return output_depth

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx
    ):
        return [x.to(device) for x in batch]

    def training_step(self, batch: Any, batch_idx: int):
        loss_info = {}
        stage = "train"
        for k, v in loss_info.items():
            if "loss" in k:
                self.log(
                    f"{stage}/{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=self.do_log_with_logger,
                )

    def validation_step(self, batch: Any, batch_idx: int):
        output_depth, ground_truth = self.eval_step(batch, batch_idx)

        mae, rmse, imae, irmse = self.calc_metrics(
            self.train_total, output_depth, ground_truth
        )
        self.log_metrics("val", mae, rmse, imae, irmse)

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {
                    "params": self.depth_net.parameters(),
                    "weight_decay": self.config.weight_decay_depth,
                }
            ],
            lr=self.config.learning_rate,
        )

    def eval_step(self, batch: Any, batch_idx: int):
        output_depth = ground_truth = None

        return output_depth, ground_truth

    def test_step(self, batch: Any, batch_idx: int):
        output_depth, ground_truth = self.eval_step(batch, batch_idx)
        mae, rmse, imae, irmse = self.calc_metrics(
            self.test_total, output_depth, ground_truth
        )
        self.log_metrics("test", mae, rmse, imae, irmse)

    def log_metrics(self, stage, mae, rmse, imae, irmse):
        self.log(
            f"{stage}/mae",
            mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=self.do_log_with_logger,
        )
        self.log(
            f"{stage}/rmse",
            rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=self.do_log_with_logger,
        )
        self.log(
            f"{stage}/imae",
            imae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=self.do_log_with_logger,
        )
        self.log(
            f"{stage}/irmse",
            irmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=self.do_log_with_logger,
        )

    def calc_metrics(self, metric_fn, output_depth, ground_truth):
        return metric_fn(output_depth, ground_truth)

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output_depth = self.eval_model_step(batch, self.val_transforms)
        return output_depth


if __name__ == "__main__":
    depth_model = DepthModel()
    transforms = {"train": None, "val": None}
    _ = UnetLitModule(depth_model, transforms=transforms, config=None)
