import argparse
import json
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision as tv
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from layout_aware_monodepth.cfg import cfg
from layout_aware_monodepth.data.monodepth import KITTIDataset, NYUv2Dataset
from layout_aware_monodepth.data.transforms import (
    ToTensor,
    test_transform,
    train_transform,
)
from layout_aware_monodepth.logging_utils import log_metric, log_params_to_exp
from layout_aware_monodepth.losses import MSELoss, SILogLoss
from layout_aware_monodepth.metrics import RunningAverageDict, calc_metrics
from layout_aware_monodepth.model import DepthModel
from layout_aware_monodepth.pipeline_utils import create_tracking_exp
from layout_aware_monodepth.postprocessing import (
    compute_eval_mask,
    postproc_eval_depths,
)
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

    def eval_step(self, model, batch, criterion):
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


def run(args):
    pl.seed_everything(1234)

    if args.ds == "kitti":
        config_path = "../configs/kitti_ds.yaml"
    else:
        config_path = "../configs/nyu_ds.yaml"

    cfg.exp_disabled = args.exp_disabled
    cfg.use_single_sample = args.use_single_sample
    cfg.do_overfit = args.do_overfit
    cfg.line_op = args.line_op
    cfg.num_epochs = args.num_epochs

    ds_args = argparse.Namespace(**yaml.load(open(config_path), Loader=yaml.FullLoader))
    for k, v in vars(args).items():
        setattr(ds_args, k, v)

    if args.ds == "kitti":
        ds_cls = KITTIDataset
    else:
        ds_cls = NYUv2Dataset

    train_ds = ds_cls(
        ds_args,
        "train",
        ds_args.split,
        transform=train_transform,
        do_augment=False,
    )

    if cfg.use_single_sample and cfg.do_overfit:
        ds_args.batch_size = 1
        cfg.num_epochs = 100
        cfg.vis_freq_epochs = 10
        ds_subset = torch.utils.data.Subset(train_ds, range(0, 1))
        train_subset = val_subset = test_subset = ds_subset
    else:
        if cfg.do_overfit:
            ds_subset = torch.utils.data.Subset(train_ds, range(0, 480))
        else:
            ds_subset = torch.utils.data.Subset(train_ds, range(0, 11_000))
        train_ds_len = int(len(ds_subset) * 0.8)
        val_ds_len = int(len(ds_subset) * 0.1)
        train_subset = torch.utils.data.Subset(ds_subset, range(0, train_ds_len))
        val_subset = torch.utils.data.Subset(
            ds_subset, range(train_ds_len, train_ds_len + val_ds_len)
        )
        if cfg.do_overfit and not args.use_eigen:
            test_subset = torch.utils.data.Subset(
                ds_subset, range(train_ds_len + val_ds_len, len(ds_subset))
            )
        else:
            test_subset = ds_cls(
                ds_args,
                "test",
                ds_args.split,
                transform=test_transform,
                do_augment=False,
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
        benchmark_batch = train_ds.load_benchmark_batch(benchmark_paths)

    model = DepthModel(
        in_channels=4 if args.line_op in ["concat", "concat_binary"] else 3,
        use_attn=args.use_attn,
        use_extra_conv=args.use_extra_conv,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = SILogLoss()

    epoch_bar = tqdm(total=cfg.num_epochs, leave=False)
    experiment = create_tracking_exp(cfg)
    exp_dir = f"{cfg.exp_base_dir}/{experiment.name}"
    os.makedirs(exp_dir, exist_ok=True)

    experiment.add_tags(
        [
            args.ds,
            "overfit" if cfg.do_overfit else "full",
            f"{cfg.line_op}_lines",
            f"filter_{args.line_filter}",
        ]
        + args.exp_tags
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_params_to_exp(
        experiment,
        cfg.params(),
        "cfg",
    )
    log_params_to_exp(
        experiment,
        vars(ds_args),
        "ds_args",
    )
    log_params_to_exp(
        experiment,
        vars(args),
        "args",
    )
    experiment.log_parameters({"model/num_params": num_params})

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

        log_metric(experiment, train_metrics_avg.get_value(), epoch, prefix="epoch")
        log_metric(experiment, val_metrics_avg.get_value(), epoch, prefix="epoch")

        is_last_epoch = epoch == cfg.num_epochs - 1
        if (epoch - 1) % cfg.vis_freq_epochs == 0 or is_last_epoch:
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
                max_depth=train_ds.max_depth,
            )
            experiment.log_figure(
                name,
                fig,
                step=epoch,
            )

            print(f"\nBENCHMARK metrics:\n{benchmark_metrics_avg}\n")

        if (
            cfg.do_save_model
            and (epoch - 1) % cfg.save_freq_epochs == 0
            or is_last_epoch
        ):
            save_path = f"{exp_dir}/model_{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            experiment.log_model(f"depth_model_{epoch}", save_path, overwrite=False)

        train_batch_bar.close()
        val_batch_bar.close()

        test_batch_bar = tqdm(total=len(test_loader), leave=True)
        test_metrics_avg = RunningAverageDict()
        test_running_losses = []
        for test_batch in test_loader:
            test_step_res = trainer.eval_step(model, test_batch, criterion)
            test_running_losses.append(test_step_res["loss"])
            test_batch_bar.update(1)
            test_metrics = {
                f"test_{k}": v for k, v in test_step_res.items() if k not in ["pred"]
            }
            test_metrics_avg.update(test_metrics)
            test_batch_bar.set_postfix(**test_metrics)

        test_batch_bar.close()

        print(f"\nTEST metrics:\n{test_metrics_avg}\n")
        log_metric(experiment, test_metrics_avg.get_value(), epoch, prefix="epoch")

    experiment.add_tags(["finished"])
    experiment.end()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="nyu", choices=["kitti", "nyu"])
    parser.add_argument("--do_overfit", action="store_true")
    parser.add_argument(
        "--line_op", choices=["overlay", "concat", "concat_binary"], default=None
    )
    parser.add_argument(
        "--line_filter", choices=["length", "vanishing_point", "length,vanishing_point"]
    )
    parser.add_argument("--use_single_sample", action="store_true")
    parser.add_argument("--exp_disabled", action="store_true")
    parser.add_argument("--use_attn", action="store_true")
    parser.add_argument("--use_extra_conv", action="store_true")
    parser.add_argument("--use_eigen", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--crop_type", choices=["garg", "eigen"], default=None)

    parser.add_argument(
        "--min_depth_eval",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--max_depth_eval", type=float, default=10)

    parser.add_argument("--exp_tags", nargs="*", default=[])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
