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


class MonodepthDataset(Dataset):
    def __init__(self, args, mode, transform=None, do_augment=False):
        self.args = args
        with open(args.filenames_file, "r") as f:
            self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.do_augment = do_augment
        self.to_tensor = ToTensor

    def __getitem__(self, idx):
        if self.mode == "train":
            image, depth_gt = self.load_img_and_depth(idx)

            sample = self.prep_train_sample(image, depth_gt)

        else:
            image = self.load_rgb(idx)
            sample = self.prep_test_sample(image)

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

    def prep_train_sample(self, image, depth_gt):
        if self.args.do_random_rotate and self.do_augment:
            random_angle = (random.random() - 0.5) * 2 * self.args.degree
            image = rotate_image(image, random_angle)
            depth_gt = rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)

        if (
            image.shape[0] != self.args.input_height
            or image.shape[1] != self.args.input_width
        ):
            print(
                f"image.shape: {image.shape} and self.args.input_height: {self.args.input_height}. doing random crop"
            )
            image, depth_gt = random_crop(
                image, depth_gt, self.args.input_height, self.args.input_width
            )
        if self.do_augment:
            image, depth_gt = train_preprocess(image, depth_gt)

        depth_gt = self.convert_depth_to_meters(depth_gt)

        sample = {
            "image": image,
            "depth": depth_gt,
        }
        return sample

    def load_img_and_depth(self, idx):
        raise NotImplementedError

    def convert_depth_to_meters(self, depth_gt):
        raise NotImplementedError

    def load_rgb(self, path):
        raise NotImplementedError

    def prep_test_sample(self, image):
        image = np.asarray(image, dtype=np.float32)
        image /= 255.0

        sample = {"image": image}
        return sample

    def __len__(self):
        return len(self.filenames)


class KITTIDataset(MonodepthDataset):
    def __init__(self, args, mode, transform=None, do_augment=False):
        super().__init__(args, mode, transform, do_augment)

    def load_img_and_depth(self, idx):
        sample_path = self.filenames[idx]
        rgb_file = sample_path.split()[0]
        image_path = os.path.join(self.args.data_path, rgb_file)

        depth_file = sample_path.split()[1]

        depth_path = os.path.join(self.args.gt_path, depth_file)

        if self.args.use_right is True:
            image_path = image_path.replace("image_02", "image_03")
            depth_path = depth_path.replace("image_02", "image_03")

        image = self.load_rgb(image_path)
        depth_gt = self.load_rgb(depth_path)

        if self.args.do_kb_crop is True:
            image, depth_gt = kb_crop(image, depth_gt)
        return image, depth_gt

    def convert_depth_to_meters(self, depth_gt):
        depth_gt = depth_gt / 256.0
        return depth_gt

    def load_rgb(self, path):
        return Image.open(path)


# image.shape: (427, 565, 3)
class NYUv2Dataset(MonodepthDataset):
    def __init__(self, args, mode, transform=None, do_augment=False):
        super().__init__(args, mode, transform, do_augment)
        import h5py

        # with h5py.File(f"{args.data_path}/nyu_depth_v2_labeled.mat", "r") as hdf:
        hdf = h5py.File(f"{args.data_path}/nyu_depth_v2_labeled.mat", "r")
        self.images = hdf.get("images")
        self.depths = hdf.get("depths")

    def load_img_and_depth(self, idx):
        image = Image.fromarray(self.images[idx].transpose(1, 2, 0))
        depth_gt = Image.fromarray(self.depths[idx], mode='F')

        # To avoid blank boundaries due to pixel registration
        image, depth_gt = self.handle_blank_boundaries(image, depth_gt)

        return image, depth_gt

    def convert_depth_to_meters(self, depth_gt):
        depth_gt = depth_gt / 4.0
        # depth_gt = depth_gt / 1000.0
        return depth_gt

    def handle_blank_boundaries(self, image, depth_gt):
        if self.args.input_height == 480:
            depth_gt = np.array(depth_gt)
            valid_mask = np.zeros_like(depth_gt)
            valid_mask[45:472, 43:608] = 1
            depth_gt[valid_mask == 0] = 0
            depth_gt = Image.fromarray(depth_gt)
        else:
            depth_gt = depth_gt.crop((43, 45, 608, 472))
            image = image.crop((43, 45, 608, 472))
        return image, depth_gt

    def load_rgb(self, idx):
        image = self.images[idx]
        image = Image.fromarray(image)
        return image
