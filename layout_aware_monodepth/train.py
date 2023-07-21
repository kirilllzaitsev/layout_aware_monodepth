import argparse

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from layout_aware_monodepth.cfg import cfg
from layout_aware_monodepth.dataset.monodepth import KITTIDataset
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
    ds_args = argparse.Namespace(
        # **yaml.load(open("../configs/kitti_ds.yaml"), Loader=yaml.FullLoader)
        **yaml.load(open("../configs/kitti_ds_overfit.yaml"), Loader=yaml.FullLoader)
    )
    # ds = NYUv2Dataset(ds_args, "train", transform=train_transform)
    ds = KITTIDataset(ds_args, ds_args.mode, transform=train_transform)

    ds_subset = torch.utils.data.Subset(ds, range(0, 1))

    train_loader = DataLoader(ds_subset, batch_size=ds_args.batch_size, shuffle=False)
    val_loader = DataLoader(ds_subset, batch_size=ds_args.batch_size)
    test_loader = DataLoader(ds_subset, batch_size=ds_args.batch_size)

    model = DepthModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    epoch_bar = tqdm(total=cfg.num_epochs, leave=False)
    experiment = create_tracking_exp(cfg)

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
                experiment.log_image(
                    torch.cat([in_depth.repeat(1, 1, 3), depth.repeat(1, 1, 3), img], dim=0),
                    f"{name}_{idx}",
                    step=epoch,
                )

        if epoch in [50]:
            torch.save(model.state_dict(), f"model_{epoch}.pth")


if __name__ == "__main__":
    run()
