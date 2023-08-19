import json
import os
import random
import re
from functools import lru_cache

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset

from layout_aware_monodepth.cfg import cfg
from layout_aware_monodepth.data.transforms import (
    ToTensor,
    interpolate_depth,
    kb_crop,
    resize_inputs,
    rotate_image,
    test_transform,
    train_preprocess,
)


class MonodepthDataset(Dataset):
    max_depth = None
    target_shape = None

    def __init__(
        self,
        args,
        mode,
        split=None,
        transform=None,
        do_augment=False,
    ):
        self.args = args

        self.mode = mode
        self.split = split
        self.transform = transform
        self.do_augment = do_augment
        self.to_tensor = ToTensor
        self.filenames = []
        self.data_dir = self.args.data_path

        if self.args.line_op is not None:
            from layout_aware_monodepth.data.tmp import load_deeplsd

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.deeplsd = load_deeplsd(self.device)

    def __getitem__(self, idx):
        image, depth_gt = self.load_img_and_depth(self.filenames[idx])

        sample = self.prep_train_sample(image, depth_gt, do_augment=self.do_augment)

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

    def load_benchmark_batch(self, sample_paths):
        images = []
        depths = []
        for paths_map in sample_paths:
            image, depth_gt = self.load_img_and_depth(paths_map)
            sample = self.prep_train_sample(image, depth_gt, do_augment=False)

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

        image, depth_gt = resize_inputs(image, depth_gt, target_shape=self.target_shape)

        depth_gt = np.expand_dims(depth_gt, axis=2)

        if do_augment:
            image, depth_gt = train_preprocess(image, depth_gt)

        depth_gt = self.convert_depth_to_meters(depth_gt)
        depth_gt = np.clip(depth_gt, 0, self.max_depth)
        depth_gt /= self.max_depth

        if self.args.line_op == "overlay":
            image = self.overlay_lines(image)
        elif self.args.line_op in ["concat", "concat_binary"]:
            image = self.concat_lines(image)

        sample = {
            "image": image,
            "depth": depth_gt,
        }
        return sample

    def prep_test_sample(self, image):
        image = np.asarray(image, dtype=np.float32)
        image /= 255.0
        image = resize_inputs(image, target_shape=self.target_shape)

        sample = {"image": image}
        return sample

    def run_line_detector(self, image):
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

        return out

    def overlay_lines(self, image):
        out = self.run_line_detector(image)
        pred_lines = out["lines"][0].astype(np.int32)

        line_thickness = 2
        overlay = image.copy()
        for line in pred_lines:
            overlay = cv2.line(
                overlay, tuple(line[0]), tuple(line[1]), (0, 1, 0), line_thickness
            )

        return overlay

    def concat_lines(self, image):
        out = self.run_line_detector(image)

        if self.args.line_op == "concat":
            df_norm = out["df_norm"][0].cpu().numpy()

            concat = (df_norm - np.min(df_norm)) / (np.max(df_norm) - np.min(df_norm))
            concat = np.expand_dims(concat, axis=2)
        elif self.args.line_op == "concat_binary":
            lines = out["lines"][0].astype(np.int32)

            if self.args.line_filter is not None:
                if "vanishing_point" in self.args.line_filter:
                    lines = self.filter_lines_by_vp(lines, out["vp_labels"][0])
                if "length" in self.args.line_filter:
                    lines = self.filter_lines_by_length(lines)

            concat = np.zeros((image.shape[0], image.shape[1], 1))
            for line in lines:
                concat = cv2.line(concat, tuple(line[0]), tuple(line[1]), (1, 1, 1), 2)
        else:
            raise NotImplementedError

        concat = concat.astype(np.float32)
        concat = np.concatenate((image, concat), axis=2)
        return concat

    def filter_lines_by_vp(self, lines, vp_labels):
        vp_labels = np.array(vp_labels)
        mask = vp_labels == -1
        lines = lines[mask]
        return lines

    def filter_lines_by_length(self, lines, min_length=10, use_min_length=False):
        line_lengths = np.sqrt(
            (lines[:, 0, 0] - lines[:, 1, 0]) ** 2
            + (lines[:, 0, 1] - lines[:, 1, 1]) ** 2
        )
        len_mean = np.mean(line_lengths)
        new_lines = []
        for idx, line in enumerate(lines):
            length = line_lengths[idx]
            if use_min_length:
                if length > min_length:
                    new_lines.append(line)
            else:
                if length > len_mean / 4:
                    new_lines.append(line)
        return np.array(new_lines)

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
    target_shape = (640, 192)

    def __init__(self, *args_, **kwargs):
        super().__init__(*args_, **kwargs)

        with open(self.args.filenames_file) as json_file:
            json_data = json.load(json_file)
            if self.split == "eigen":
                self.filenames = json_data["eigen"][self.mode]
            else:
                self.filenames = json_data[self.mode]

    def load_img_and_depth(self, paths_map):
        rgb_path = (
            self.from_local_path_to_cluster(paths_map["rgb"])
            if cfg.is_cluster
            else paths_map["rgb"]
        )
        if self.split == "eigen":
            if "data_" in rgb_path:
                image_path = os.path.join(
                    self.data_dir,
                    rgb_path,
                )
                depth_path = os.path.join(
                    self.data_dir,
                    paths_map["gt"],
                )
            else:
                image_path = os.path.join(
                    self.data_dir.replace("/kitti-depth", ""),
                    "kitti_raw_data",
                    rgb_path,
                )
                for mode in ["train", "val"]:
                    depth_path = os.path.join(
                        self.data_dir,
                        f"data_depth_annotated/{mode}",
                        paths_map["gt"],
                    )
                    if os.path.exists(depth_path):
                        break
                else:
                    raise ValueError(f"depth_path not found: {paths_map['gt']}")
        elif "data_" in rgb_path:
            image_path = os.path.join(self.data_dir, rgb_path)
            depth_path = os.path.join(
                self.data_dir, self.from_local_path_to_cluster(paths_map["gt"])
            )
        else:
            if cfg.is_cluster:
                image_path = os.path.join(
                    self.data_dir,
                    "data_rgb",
                    rgb_path,
                )
            else:
                image_path = os.path.join(self.data_dir, "data_rgb", rgb_path)
            depth_path = os.path.join(
                self.data_dir, "data_depth_annotated", paths_map["gt"]
            )

        return self.load_img_and_depth_from_path(image_path, depth_path)

    def from_local_path_to_cluster(self, path):
        # example: train/2011_09_26_drive_0002_sync
        sync_folder = path[path.find("/") + 1 :]
        day = sync_folder[:10]
        if re.match(r"\d{4}_\d{2}_\d{2}", day) is not None:
            return day + "/" + sync_folder
        return path

    def load_img_and_depth_from_path(self, image_path, depth_path):
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


class NYUv2Dataset(MonodepthDataset):
    max_depth = 10.0
    target_shape = (256, 256)

    def __init__(self, *args_, do_interpolate_depth=False, do_crop=True, **kwargs):
        super().__init__(*args_, **kwargs)
        self.do_interpolate_depth = do_interpolate_depth
        self.do_crop = do_crop
        with open(self.args.filenames_file) as json_file:
            json_data = json.load(json_file)
            self.filenames = json_data[self.mode]

    def load_img_and_depth(self, paths_map):
        path_file = os.path.join(self.data_dir, paths_map["filename"])

        return self.load_img_and_depth_from_path(path_file)

    def load_img_and_depth_from_path(self, path_file):
        f = h5py.File(path_file, "r")
        image = self._load_rgb(f)

        dep_h5 = f["depth"][:]
        if self.do_interpolate_depth:
            depth_gt = interpolate_depth(dep_h5.squeeze(), do_multiscale=True)
        else:
            depth_gt = dep_h5.squeeze()
        depth_gt = Image.fromarray(depth_gt.astype("float32"), mode="F")
        if self.do_crop:
            depth_gt = depth_gt.crop((43, 45, 608, 472))

        return image, depth_gt

    def convert_depth_to_meters(self, depth_gt):
        depth_gt = depth_gt / 1.0
        # depth_gt = depth_gt / 4.0  # original .mat
        # depth_gt = depth_gt / 1000.0
        return depth_gt

    def load_rgb(self, idx):
        path_file = os.path.join(self.data_dir, self.filenames[idx]["filename"])

        f = h5py.File(path_file, "r")
        image = self._load_rgb(f)
        return image

    def _load_rgb(self, f):
        rgb_h5 = f["rgb"][:].transpose(1, 2, 0)
        image = Image.fromarray(rgb_h5, mode="RGB")
        if self.do_crop:
            image = image.crop((43, 45, 608, 472))
        return image
