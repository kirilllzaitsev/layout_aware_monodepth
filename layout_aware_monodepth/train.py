import argparse

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
    parser.add_argument("--exp_disabled", action="store_true")
    parser.add_argument("--do_overlay_lines", action="store_true")
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

    ds_args = argparse.Namespace(**yaml.load(open(config_path), Loader=yaml.FullLoader))

    if args.ds == "kitti":
        ds_cls = KITTIDataset
    else:
        ds_cls = NYUv2Dataset

    ds = ds_cls(
        ds_args,
        ds_args.mode,
        transform=train_transform,
        do_overlay_lines=args.do_overlay_lines,
    )

    if args.do_overfit:
        ds_subset = torch.utils.data.Subset(ds, range(0, 1))
    else:
        ds_subset = ds

    train_loader = DataLoader(ds_subset, batch_size=ds_args.batch_size, shuffle=False)
    val_loader = DataLoader(ds_subset, batch_size=ds_args.batch_size)
    test_loader = DataLoader(ds_subset, batch_size=ds_args.batch_size)

    model = DepthModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    epoch_bar = tqdm(total=cfg.num_epochs, leave=False)
    experiment = create_tracking_exp(cfg)

    experiment.add_tags(
        [
            args.ds,
            "overfit" if args.do_overfit else "full",
            "overlay" if args.do_overlay_lines else "no_overlay",
        ]
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
            out = train_step_res["pred"].detach().cpu().permute(0, 2, 3, 1)
            images = train_batch["image"].detach().cpu().permute(0, 2, 3, 1)
            depths = train_batch["depth"].detach().cpu()

            name = "preds/depth"
            for idx in range(len(out)):
                experiment.log_image(
                    out[idx].numpy(),
                    f"{name}_{idx}",
                    step=epoch,
                )

            name = "preds/sample"
            for idx, (img, in_depth, depth) in enumerate(zip(images, depths, out)):
                concat_sample = torch.cat(
                    [in_depth.repeat(1, 1, 3), depth.repeat(1, 1, 3), img],
                    dim=0,
                )

                if args.ds == "nyu":
                    target_shape = [
                        concat_sample.shape[0] // 2,
                        concat_sample.shape[1] // 2,
                    ]
                    concat_sample = tv.transforms.Resize(target_shape, antialias=True)(
                        concat_sample.permute(2, 0, 1)
                    ).permute(1, 2, 0)
                experiment.log_image(
                    concat_sample,
                    f"{name}_{idx}",
                    step=epoch,
                )

        if epoch in [50] and cfg.do_save_model:
            torch.save(model.state_dict(), f"model_{epoch}.pth")

    experiment.add_tags(["finished"])


if __name__ == "__main__":
    run()
