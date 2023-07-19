import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from layout_aware_monodepth.dataset.transforms import (
    ToTensor,
    kb_crop,
    random_crop,
    rotate_image,
    train_preprocess,
)


def preprocessing_transforms(mode):
    return transforms.Compose([ToTensor(mode=mode)])


class KITTIDataset(Dataset):
    def __init__(self, args, mode, transform=None):
        self.args = args
        with open(args.filenames_file, "r") as f:
            self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        if self.mode == "train":
            if self.args.dataset == "kitti":
                rgb_file = sample_path.split()[0]
                # depth_file = os.path.join(sample_path.split()[0].split('/')[0], sample_path.split()[1])
                depth_file = sample_path.split()[1]

                if self.args.use_right is True and random.random() > 0.5:
                    rgb_file.replace("image_02", "image_03")
                    depth_file.replace("image_02", "image_03")
            else:
                rgb_file = sample_path.split()[0]
                depth_file = sample_path.split()[1]

            image_path = os.path.join(
                self.args.data_path, "./" + sample_path.split()[0]
            )
            depth_path = os.path.join(self.args.gt_path, "./" + sample_path.split()[1])
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)

            if self.args.do_kb_crop is True:
                image, depth_gt = kb_crop(image, depth_gt)

            # To avoid blank boundaries due to pixel registration
            if self.args.dataset == "nyu":
                if self.args.input_height == 480:
                    depth_gt = np.array(depth_gt)
                    valid_mask = np.zeros_like(depth_gt)
                    valid_mask[45:472, 43:608] = 1
                    depth_gt[valid_mask == 0] = 0
                    depth_gt = Image.fromarray(depth_gt)
                else:
                    depth_gt = depth_gt.crop((43, 45, 608, 472))
                    image = image.crop((43, 45, 608, 472))

            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = rotate_image(image, random_angle)
                depth_gt = rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.args.dataset == "nyu":
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0

            if (
                image.shape[0] != self.args.input_height
                or image.shape[1] != self.args.input_width
            ):
                image, depth_gt = random_crop(
                    image, depth_gt, self.args.input_height, self.args.input_width
                )
            image, depth_gt = train_preprocess(image, depth_gt)
            sample = {
                "image": image,
                "depth": depth_gt,
            }

        else:
            data_path = self.args.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            sample = {"image": image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    from tmp import parser
    args = parser.parse_args()
    ds = KITTIDataset(args, "train", transform=transforms.Compose([ToTensor("train")]))
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    x = ds[0]
    print(x)
    x = next(iter(dl))
    print(x)
