from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from layout_aware_monodepth.dataset.monodepth import KITTIDataset
from layout_aware_monodepth.model import DepthModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LitModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DepthModel()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        depth = self.model(x)
        return depth

    def training_step(self, batch, batch_idx):
        x, y = batch["image"].to(device), batch["depth"].permute(0, 3, 1, 2).to(device)
        out = self.model(x)
        loss = F.mse_loss(out, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    import argparse

    import yaml

    from layout_aware_monodepth.dataset.transforms import ToTensor, train_transform

    ds_args = argparse.Namespace(
        **yaml.load(open("../configs/kitti_ds.yaml"), Loader=yaml.FullLoader)
    )
    # ds = NYUv2Dataset(ds_args, "train", transform=train_transform)
    ds = KITTIDataset(ds_args, "train", transform=train_transform)

    train_loader = DataLoader(ds, batch_size=args.batch_size)
    val_loader = DataLoader(ds, batch_size=args.batch_size)
    test_loader = DataLoader(ds, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = LitModule()

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, accelerator="gpu", gpus=1)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == "__main__":
    cli_main()
