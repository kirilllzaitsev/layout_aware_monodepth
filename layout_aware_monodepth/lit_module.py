from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from layout_aware_monodepth.dataset.monodepth import KITTIDataset
from layout_aware_monodepth.model import DepthModel


def comet_image(exp, img, name, **kwargs):
    # img: B, H, W, C
    for i, frame in enumerate(img.detach().cpu()):
        exp.log_image(frame, name=f"{name}_{i}", **kwargs)


class LitModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DepthModel()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        depth = self.model(x)
        return depth

    def training_step(self, batch, batch_idx):
        x, y = batch["image"].to(self.device), batch["depth"].permute(0, 3, 1, 2).to(
            self.device
        )
        out = self.model(x)
        loss = F.mse_loss(out, y)
        self.log_dict({"loss": loss, "log": {"train_loss": loss}})

        exp = self.logger.experiment
        comet_image(exp, out, name="train/out", step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def cli_main():
    pl.seed_everything(1234)

    import argparse

    import yaml

    from layout_aware_monodepth.dataset.transforms import ToTensor, train_transform

    ds_args = argparse.Namespace(
        **yaml.load(open("../configs/kitti_ds.yaml"), Loader=yaml.FullLoader)
    )
    # ds = NYUv2Dataset(ds_args, "train", transform=train_transform)
    ds = KITTIDataset(ds_args, "train", transform=train_transform)

    ds_args.batch_size = 1

    train_loader = DataLoader(ds, batch_size=ds_args.batch_size, shuffle=False)
    val_loader = DataLoader(ds, batch_size=ds_args.batch_size)
    test_loader = DataLoader(ds, batch_size=ds_args.batch_size)

    # ------------
    # model
    # ------------
    model = LitModule()

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
        accelerator="gpu",
        gpus=1,
        # fast_dev_run=True,
        overfit_batches=1,
        max_epochs=100,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    # result = trainer.test(test_dataloaders=test_loader)
    # print(result)


if __name__ == "__main__":
    cli_main()
