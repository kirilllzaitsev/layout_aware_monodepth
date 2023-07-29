import argparse
import json

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision as tv
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from layout_aware_monodepth.cfg import cfg
from layout_aware_monodepth.data.monodepth import KITTIDataset, NYUv2Dataset
from layout_aware_monodepth.data.transforms import ToTensor, train_transform
from layout_aware_monodepth.losses import MSELoss, SILogLoss
from layout_aware_monodepth.metrics import calc_metrics
from layout_aware_monodepth.model import DepthModel
from layout_aware_monodepth.pipeline_utils import create_tracking_exp
from layout_aware_monodepth.postprocessing import (
    compute_eval_mask,
    postproc_eval_depths,
)
from layout_aware_monodepth.logging_utils import log_metric
from layout_aware_monodepth.vis_utils import plot_samples_and_preds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self, args, model, optimizer, criterion, train_loader, val_loader, test_loader
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args

    def train_step(self, model, batch, criterion, optimizer):
        model.train()
        x, y = batch["image"].to(device), batch["depth"].permute(0, 3, 1, 2).to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "pred": out}

    def test_step(self, model, batch, criterion):
        model.eval()
        result = {}
        with torch.no_grad():
            x, y = batch["image"].to(device), batch["depth"].permute(0, 3, 1, 2).to(
                device
            )
            pred = model(x)
            test_loss = criterion(pred, y)
            result["loss"] = test_loss.item()
            result["pred"] = pred
            pred, y = postproc_eval_depths(
                pred,
                y,
                min_depth=self.args.min_depth_eval,
                max_depth=self.args.max_depth_eval,
            )
            eval_mask = compute_eval_mask(
                y,
                min_depth=self.args.min_depth_eval,
                max_depth=self.args.max_depth_eval,
                crop_type=self.args.crop_type,
                ds_name=self.args.ds,
            )
            metrics = calc_metrics(pred, y, mask=eval_mask)
            return {**result, **metrics}


def run():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="nyu", choices=["kitti", "nyu"])
    parser.add_argument("--do_overfit", action="store_true")
    parser.add_argument("--do_overlay_lines", action="store_true")
    parser.add_argument("--use_single_sample", action="store_true")
    parser.add_argument("--exp_disabled", action="store_true")
    parser.add_argument("--crop_type", choices=["garg", "eigen"], default=None)

    parser.add_argument(
        "--min_depth_eval",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--max_depth_eval", type=float, default=10)

    parser.add_argument("--exp_tags", nargs="+", default=[])
    args = parser.parse_args()

    if args.ds == "kitti":
        if args.do_overfit:
            config_path = "../configs/kitti_ds_overfit.yaml"
        else:
            config_path = "../configs/kitti_ds.yaml"
    else:
        if args.do_overfit:
            config_path = "../configs/nyu_ds_overfit.yaml"
        else:
            config_path = "../configs/nyu_ds.yaml"

    cfg.exp_disabled = args.exp_disabled
    cfg.use_single_sample = args.use_single_sample
    cfg.do_overfit = args.do_overfit
    cfg.do_overlay_lines = args.do_overlay_lines

    ds_args = argparse.Namespace(**yaml.load(open(config_path), Loader=yaml.FullLoader))

    if args.ds == "kitti":
        ds_cls = KITTIDataset
    else:
        ds_cls = NYUv2Dataset

    ds = ds_cls(
        ds_args,
        ds_args.mode,
        transform=train_transform,
        do_overlay_lines=cfg.do_overlay_lines,
    )

    if cfg.use_single_sample and cfg.do_overfit:
        ds_args.batch_size = 1
        cfg.num_epochs = 100
        cfg.vis_freq_epochs = 10
        ds_subset = torch.utils.data.Subset(ds, range(0, 1))
        train_subset = val_subset = test_subset = ds_subset
    else:
        if cfg.do_overfit:
            ds_subset = torch.utils.data.Subset(ds, range(0, 280))
        else:
            ds_subset = ds
        train_ds_len = int(len(ds_subset) * 0.8)
        val_ds_len = int(len(ds_subset) * 0.1)
        train_subset = torch.utils.data.Subset(ds_subset, range(0, train_ds_len))
        val_subset = torch.utils.data.Subset(
            ds_subset, range(train_ds_len, train_ds_len + val_ds_len)
        )
        test_subset = torch.utils.data.Subset(
            ds_subset, range(train_ds_len + val_ds_len, len(ds_subset))
        )

    train_loader = DataLoader(train_subset, batch_size=ds_args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=ds_args.batch_size)
    test_loader = DataLoader(test_subset, batch_size=ds_args.batch_size)

    if cfg.use_single_sample:
        benchmark_batch = next(iter(train_loader))
    else:
        benchmark_paths = json.load(open("../data/data_splits/eval_samples.json"))[
            args.ds
        ]
        benchmark_batch = ds.load_benchmark_batch(benchmark_paths)

    model = DepthModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = SILogLoss()

    epoch_bar = tqdm(total=cfg.num_epochs, leave=False)
    experiment = create_tracking_exp(cfg)

    experiment.add_tags(
        [
            args.ds,
            "overfit" if cfg.do_overfit else "full",
            "overlay" if cfg.do_overlay_lines else "no_overlay",
        ]
        + args.exp_tags
    )

    global_step = 0

    trainer = Trainer(
        args, model, optimizer, criterion, train_loader, val_loader, test_loader
    )

    for epoch in range(cfg.num_epochs):
        train_batch_bar = tqdm(total=len(train_loader), leave=True)
        val_batch_bar = tqdm(total=len(val_loader), leave=True)

        train_metrics_avg = RunningAverageDict()
        val_metrics_avg = RunningAverageDict()
        benchmark_metrics_avg = RunningAverageDict()

        epoch_bar.set_description(f"Epoch {epoch}")

        train_running_losses = []
        val_running_losses = []

        for train_batch in train_loader:
            train_step_res = trainer.train_step(
                model, train_batch, criterion, optimizer
            )
            loss = train_step_res["loss"]
            train_running_losses.append(loss)
            train_batch_bar.update(1)
            train_metrics = {
                f"train_{k}": v for k, v in train_step_res.items() if k not in ["pred"]
            }

            train_batch_bar.set_postfix(**train_metrics)
            global_step += 1
            log_metric(experiment, train_metrics, global_step, prefix="step")
            train_metrics_avg.update(train_metrics)

        for val_batch in val_loader:
            val_step_res = trainer.eval_step(model, val_batch, criterion)
            val_running_losses.append(val_step_res["loss"])
            val_batch_bar.update(1)
            val_metrics = {
                f"val_{k}": v for k, v in val_step_res.items() if k not in ["pred"]
            }
            val_batch_bar.set_postfix(**val_metrics)
            val_metrics_avg.update(val_metrics)

        epoch_bar.update(1)

        print(f"\nTRAIN metrics:\n{train_metrics_avg}\n")
        print(f"\nVAL metrics:\n{val_metrics_avg}\n")

        experiment.log_metric(
            "epoch/train_loss",
            train_metrics_avg.get_value()["avg_train_loss"],
            step=epoch,
        )
        experiment.log_metric(
            "epoch/val_loss",
            val_metrics_avg.get_value()["avg_val_loss"],
            step=epoch,
        )

        if (epoch - 1) % cfg.vis_freq_epochs == 0 or epoch == cfg.num_epochs - 1:
            benchmark_step_res = trainer.eval_step(model, benchmark_batch, criterion)
            benchmark_metrics = {
                f"benchmark_{k}": v
                for k, v in benchmark_step_res.items()
                if k not in ["pred"]
            }
            benchmark_metrics_avg.update(benchmark_metrics)
            log_metric(experiment, benchmark_metrics, epoch, prefix="epoch")
            out = benchmark_step_res["pred"].detach().cpu().permute(0, 2, 3, 1)

            name = "preds/depth"
            for idx in range(len(out)):
                experiment.log_image(
                    out[idx].numpy(),
                    f"{name}_{idx}",
                    step=epoch,
                )

            name = "preds/sample"
            fig = plot_samples_and_preds(
                benchmark_batch,
                out,
                with_depth_diff=True,
                with_colorbar=True,
                max_depth=ds.max_depth,
            )
            experiment.log_figure(
                name,
                fig,
                step=epoch,
            )

        if epoch in [50] and cfg.do_save_model:
            torch.save(model.state_dict(), f"model_{epoch}.pth")

        train_batch_bar.close()
        val_batch_bar.close()

    test_batch_bar = tqdm(total=len(test_loader), leave=True)
    test_running_losses = []
    for test_batch in test_loader:
        test_step_res = trainer.test_step(model, test_batch, criterion)
        test_running_losses.append(test_step_res["loss"])
        test_batch_bar.update(1)
        test_metrics = {
            f"test_{k}": v for k, v in test_step_res.items() if k not in ["pred"]
        }
        test_batch_bar.set_postfix(**test_metrics)

    test_batch_bar.close()
    avg_test_loss = sum(test_running_losses) / len(test_running_losses)

    experiment.log_metric(
        "epoch/test_loss",
        avg_test_loss,
        step=epoch,
    )

    experiment.add_tags(["finished"])


if __name__ == "__main__":
    run()
