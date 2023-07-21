import torch
import torch.nn.functional as F
from tqdm import tqdm

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
    return {"loss": loss}


# define model testing step
def test_step(model, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch["image"].to(device), batch["depth"].permute(0, 3, 1, 2).to(device)
        out = model(x)
        test_loss = calculate_loss(out, y)
        return {"loss": test_loss, "out": out}


def calculate_loss(input1, input2):
    loss = F.mse_loss(input1, input2)
    return loss


def run():
    import argparse

    import pytorch_lightning as pl
    import yaml
    from torch.utils.data import DataLoader

    from layout_aware_monodepth.cfg import cfg
    from layout_aware_monodepth.dataset.monodepth import KITTIDataset
    from layout_aware_monodepth.dataset.transforms import ToTensor, train_transform
    from layout_aware_monodepth.model import DepthModel

    pl.seed_everything(1234)
    ds_args = argparse.Namespace(
        **yaml.load(open("../configs/kitti_ds.yaml"), Loader=yaml.FullLoader)
    )
    # ds = NYUv2Dataset(ds_args, "train", transform=train_transform)
    ds = KITTIDataset(ds_args, "train", transform=train_transform)
    ds_args.batch_size = 8

    ds_subset = torch.utils.data.Subset(ds, range(0, 100))

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

    for epoch in range(cfg.num_epochs):
        train_batch_bar = tqdm(total=len(train_loader), leave=True)
        test_batch_bar = tqdm(total=len(test_loader), leave=True)
        epoch_bar.set_description(f"Epoch {epoch}")

        train_running_losses = []
        test_running_losses = []

        for batch in train_loader:
            train_step_res = train_step(model, batch, criterion, optimizer)
            train_running_losses.append(train_step_res["loss"].item())
            train_batch_bar.update(1)
            train_batch_bar.set_postfix(**{"train_loss": train_step_res["loss"].item()})
            global_step += 1
            experiment.log_metric(
                "step/train_loss",
                avg_train_loss,
                step=global_step,
            )

        for batch in test_loader:
            test_step_res = test_step(model, batch)
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


        if epoch in [50]:
            torch.save(model.state_dict(), f"model_{epoch}.pth")


if __name__ == "__main__":
    run()
