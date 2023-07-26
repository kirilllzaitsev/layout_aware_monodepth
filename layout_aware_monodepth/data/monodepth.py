import json
import os
import random
from functools import lru_cache

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from layout_aware_monodepth.data.transforms import (
    ToTensor,
    interpolate_depth_depth,
    kb_crop,
    random_crop,
    rotate_image,
    test_transform,
    train_preprocess,
)


def preprocessing_transforms(mode):
    return transforms.Compose([ToTensor(mode=mode)])


class MonodepthDataset(Dataset):
    max_depth = None

    def __init__(
        self, args, mode, transform=None, do_augment=False, do_overlay_lines=False
    ):
        self.args = args

        self.mode = mode
        self.transform = transform
        self.do_augment = do_augment
        self.do_overlay_lines = do_overlay_lines
        self.to_tensor = ToTensor
        self.filenames = []

        if self.do_overlay_lines:
            from layout_aware_monodepth.data.tmp import load_deeplsd

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.deeplsd = load_deeplsd(self.device)

    @lru_cache(maxsize=128)
    def __getitem__(self, idx):
        if self.mode == "train":
            image, depth_gt = self.load_img_and_depth(self.filenames[idx])

            sample = self.prep_train_sample(image, depth_gt, do_augment=self.do_augment)

        else:
            image = self.load_rgb(idx)
            sample = self.prep_test_sample(image)

        if self.do_overlay_lines:
            sample["image"] = self.overlay_lines(sample["image"])

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

    def load_benchmark_batch(self, sample_paths):
        images = []
        depths = []
        for paths_map in sample_paths:
            image, depth_gt = self.load_img_and_depth(paths_map)
            sample = self.prep_train_sample(image, depth_gt, do_augment=False)

            if self.do_overlay_lines:
                sample["image"] = self.overlay_lines(sample["image"])
            sample["image"] = test_transform(sample["image"])

            images.append(sample["image"])
            depths.append(torch.from_numpy(sample["depth"]))
        return {"image": torch.stack(images), "depth": torch.stack(depths)}

    def prep_train_sample(self, image, depth_gt, do_augment=False):
        if self.args.do_random_rotate and do_augment:
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
        if do_augment:
            image, depth_gt = train_preprocess(image, depth_gt)

        depth_gt = self.convert_depth_to_meters(depth_gt)
        depth_gt /= self.max_depth

        sample = {
            "image": image,
            "depth": depth_gt,
        }
        return sample

    def prep_test_sample(self, image):
        image = np.asarray(image, dtype=np.float32)
        image /= 255.0

        sample = {"image": image}
        return sample

    def overlay_lines(self, image):
        # Detect (and optionally refine) the lines
        line_detector_img = image.copy()
        if np.max(image) <= 1.0:
            line_detector_img = line_detector_img * 255.0
        inputs = {
            "image": torch.tensor(
                cv2.cvtColor(line_detector_img, cv2.COLOR_RGB2GRAY),
                dtype=torch.float,
                device=self.device,
            )[None, None]
            / 255.0
        }
        with torch.no_grad():
            out = self.deeplsd(inputs)
            pred_lines = out["lines"][0].astype(np.int32)

        line_thickness = 2
        overlay = image.copy()
        for line in pred_lines:
            overlay = cv2.line(
                overlay, tuple(line[0]), tuple(line[1]), (0, 1, 0), line_thickness
            )

        return overlay

    def load_img_and_depth(self, paths_map):
        raise NotImplementedError

    def convert_depth_to_meters(self, depth_gt):
        raise NotImplementedError

    def load_rgb(self, path):
        raise NotImplementedError

    def __len__(self):
        return len(self.filenames)


class KITTIDataset(MonodepthDataset):
    max_depth = 80.0

    def __init__(self, *args_, use_eigen=False, **kwargs):
        self.use_eigen = use_eigen
        super().__init__(*args_, **kwargs)

        with open(self.args.filenames_file) as json_file:
            json_data = json.load(json_file)
            if use_eigen:
                self.filenames = json_data['eigen'][self.mode]
                self.args.data_path = self.args.data_path.replace('/kitti-depth', '')
            else:
                self.filenames = json_data[self.mode]


    def load_img_and_depth(self, paths_map):
        if self.use_eigen:
            image_path = os.path.join(
                self.args.data_path, "kitti_raw_data", paths_map["rgb"]
            )
            depth_path = os.path.join(
                self.args.data_path,
                f"kitti_depth_completion/train_val_split/ground_truth/{self.mode}",
                paths_map["gt"],
            )
        elif "data_" in paths_map["rgb"]:
            image_path = os.path.join(self.args.data_path, paths_map["rgb"])
            depth_path = os.path.join(self.args.data_path, paths_map["gt"])
        else:
            image_path = os.path.join(self.args.data_path, "data_rgb", paths_map["rgb"])
            depth_path = os.path.join(
                self.args.data_path, "data_depth_annotated", paths_map["gt"]
            )

        return self.load_img_and_depth_from_path(image_path, depth_path)

    def load_img_and_depth_from_path(self, image_path, depth_path):
        image = self.load_rgb(image_path)
        depth_gt = self.load_rgb(depth_path)

        if self.args.do_kb_crop is True:
            image, depth_gt = kb_crop(image, depth_gt)
        return image, depth_gt

    def convert_depth_to_meters(self, depth_gt):
        depth_gt = depth_gt / 256.0
        depth_gt = np.clip(depth_gt, 0, self.max_depth)
        return depth_gt

    def load_rgb(self, path):
        return Image.open(path)


class NYUv2Dataset(MonodepthDataset):
    max_depth = 10.0

    def __init__(self, *args_, **kwargs):
        super().__init__(*args_, **kwargs)
        with open(self.args.filenames_file) as json_file:
            json_data = json.load(json_file)
            self.filenames = json_data[self.mode]

    def load_img_and_depth(self, paths_map):
        path_file = os.path.join(self.args.data_path, paths_map["filename"])

        return self.load_img_and_depth_from_path(path_file)

    def load_img_and_depth_from_path(self, path_file):
        f = h5py.File(path_file, "r")
        image = self._load_rgb(f)

        dep_h5 = f["depth"][:]
        depth_gt = interpolate_depth_depth(dep_h5.squeeze(), do_multiscale=True)
        depth_gt = Image.fromarray(depth_gt.astype("float32"), mode="F")

        return image, depth_gt

    def convert_depth_to_meters(self, depth_gt):
        depth_gt = depth_gt / 1.0
        # depth_gt = depth_gt / 4.0  # original .mat
        # depth_gt = depth_gt / 1000.0
        return depth_gt

    def load_rgb(self, idx):
        path_file = os.path.join(self.args.data_path, self.filenames[idx]["filename"])

        f = h5py.File(path_file, "r")
        image = self._load_rgb(f)
        return image

    @classmethod
    def _load_rgb(cls, f):
        rgb_h5 = f["rgb"][:].transpose(1, 2, 0)
        image = Image.fromarray(rgb_h5, mode="RGB")
        return image
