from typing import Any, Dict, Optional, Tuple

import hydra
import omegaconf as oc
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split

from layout_aware_monodepth.data.components.data_transforms import (
    test_transform,
    train_transform,
)
from layout_aware_monodepth.data.kitti.kitti_dataset import (
    CustomKittiDCDataset,
    KittiDCDataset,
)
from layout_aware_monodepth.data.monodepth import KITTIDataset


class KittiDataModule(LightningDataModule):
    def __init__(
        self,
        ds_config: Any = None,
        data_dir: str = "data/",
        train_val_split_frac: Tuple[float, float] = (0.9, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def input_img_size(self):
        return (352, 1216)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if not self.data_train and not self.data_val and not self.data_test:
            ds_config: Any = self.hparams.ds_config

            trainset = KITTIDataset(
                ds_config,
                "train",
                transform=train_transform,
                do_overlay_lines=ds_config.do_overlay_lines,
            )

            train_len = int(len(trainset) * self.hparams.train_val_split_frac[0])
            val_len = len(trainset) - train_len
            lengths = [train_len, val_len]
            self.data_train, self.data_val = random_split(
                dataset=trainset,
                lengths=lengths,
                generator=torch.Generator().manual_seed(42),
            )

            valset = KITTIDataset(
                ds_config,
                "val",
                transform=test_transform,
                do_overlay_lines=ds_config.do_overlay_lines,
            )

            test_len = int(len(valset) * 0.3)
            val_len = len(valset) - test_len
            lengths = [test_len, val_len]
            self.data_test, self.data_val = random_split(
                dataset=valset,
                lengths=lengths,
                generator=torch.Generator().manual_seed(42),
            )

            self.data_predict = KITTIDataset(
                ds_config,
                ds_config.mode,
                transform=test_transform,
                do_overlay_lines=ds_config.do_overlay_lines,
            )

    def get_dataloader(self, dataset, shuffle: bool = True):
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.get_dataloader(self.data_train)

    def val_dataloader(self):
        return self.get_dataloader(self.data_val, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(self.data_test, shuffle=False)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    ds = hydra.utils.instantiate(cfg.data)
    ds.setup()
    print(len(ds.data_train))
    print(len(ds.data_val))


if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    main()
