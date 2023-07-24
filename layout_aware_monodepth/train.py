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
from layout_aware_monodepth.dataset.monodepth import KITTIDataset, NYUv2Dataset
from layout_aware_monodepth.dataset.transforms import ToTensor, train_transform
from layout_aware_monodepth.model import DepthModel
from layout_aware_monodepth.pipeline_utils import create_tracking_exp
from layout_aware_monodepth.vis_utils import plot_samples_and_preds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define model training step
def train_step(model, batch, criterion, optimizer):
    model.train()
    x, y = batch["image"].to(device), batch["depth"].permute(0, 3, 1, 2).to(device)
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return {"loss": loss, "pred": out}


# define model testing step
def test_step(model, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch["image"].to(device), batch["depth"].permute(0, 3, 1, 2).to(device)
        pred = model(x)
        test_loss = calculate_loss(pred, y)
        return {"loss": test_loss, "pred": pred}


def calculate_loss(input1, input2):
    loss = F.mse_loss(input1, input2)
    return loss


def run():
    pl.seed_everything(1234)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="kitti", choices=["kitti", "nyu"])
    parser.add_argument("--do_overfit", action="store_true")
    parser.add_argument("--do_overlay_lines", action="store_true")
    parser.add_argument("--use_single_sample", action="store_true")
    parser.add_argument("--exp_disabled", action="store_true")
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

    train_loader = DataLoader(
        train_subset, batch_size=ds_args.batch_size, shuffle=False
    )
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
    criterion = torch.nn.MSELoss()

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

    train_batch_bar = tqdm(total=len(train_loader), leave=True)
    test_batch_bar = tqdm(total=len(test_loader), leave=True)
    for epoch in range(cfg.num_epochs):
        epoch_bar.set_description(f"Epoch {epoch}")

        train_running_losses = []
        test_running_losses = []

        for train_batch in train_loader:
            train_step_res = train_step(model, train_batch, criterion, optimizer)
            loss = train_step_res["loss"].item()
            train_running_losses.append(loss)
            train_batch_bar.update(1)
            train_batch_bar.set_postfix(**{"train_loss": loss})
            global_step += 1
            experiment.log_metric(
                "step/train_loss",
                loss,
                step=global_step,
            )

        for test_batch in test_loader:
            test_step_res = test_step(model, test_batch)
            test_running_losses.append(test_step_res["loss"].item())
            test_batch_bar.update(1)
            test_batch_bar.set_postfix(**{"test_loss": test_step_res["loss"].item()})

        avg_train_loss = sum(train_running_losses) / len(train_running_losses)
        avg_test_loss = sum(test_running_losses) / len(test_running_losses)

        epoch_bar.update(1)
        epoch_bar.set_postfix(
            **{
                "avg_train_loss": avg_train_loss,
                "avg_test_loss": avg_test_loss,
            }
        )

        experiment.log_metric(
            "epoch/train_loss",
            avg_train_loss,
            step=epoch,
        )
        experiment.log_metric(
            "epoch/test_loss",
            avg_test_loss,
            step=epoch,
        )

        if (epoch - 1) % cfg.vis_freq_epochs == 0 or epoch == cfg.num_epochs - 1:
            benchmark_step_res = test_step(model, benchmark_batch)
            out = benchmark_step_res["pred"].detach().cpu().permute(0, 2, 3, 1)

            experiment.log_metric(
                "epoch/benchmark_loss",
                benchmark_step_res["loss"].item(),
                step=epoch,
            )

            name = "preds/depth"
            for idx in range(len(out)):
                experiment.log_image(
                    out[idx].numpy(),
                    f"{name}_{idx}",
                    step=epoch,
                )

            name = "preds/sample"
            fig = plot_samples_and_preds(benchmark_batch, out)
            experiment.log_figure(
                name,
                fig,
                step=epoch,
            )

        if epoch in [50] and cfg.do_save_model:
            torch.save(model.state_dict(), f"model_{epoch}.pth")

    experiment.add_tags(["finished"])


if __name__ == "__main__":
    run()
