import json
import os
import random
import re
from pathlib import Path

import cv2
import h5py
import numpy as np
import skimage
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
from layout_aware_monodepth.line_utils import rescale_lines
from layout_aware_monodepth.ssl_utils import get_adj_img_paths


class MonodepthDataset(Dataset):
    max_depth = None

    def __init__(
        self,
        args,
        mode,
        split=None,
        transform=None,
        do_augment=False,
        include_sample_paths=False,
    ):
        self.args = args

        self.mode = mode
        self.split = split
        self.transform = transform
        self.do_augment = do_augment
        self.target_shape = self.args.target_shape
        self.to_tensor = ToTensor
        self.filenames = []
        self.data_dir = self.args.data_path
        self.include_sample_paths = include_sample_paths

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.args.line_op is not None or getattr(self.args, "include_lines", False):
            from layout_aware_monodepth.line_utils import load_deeplsd

            self.deeplsd = load_deeplsd().to(self.device)
        self.intrinsics = torch.load(f"{self.data_dir}/intrinsics.pth")

    def __getitem__(self, idx):
        paths_map = self.filenames[idx]
        lines = (
            self.load_line_detector_res(paths_map) if self.args.do_load_lines else None
        )

        sample = self.prep_train_sample(
            paths_map, do_augment=self.do_augment, line_detector_res=lines
        )

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        if self.include_sample_paths:
            sample["rgb_path"] = self.load_paths(paths_map)[0]

        return sample

    def load_line_detector_res(self, paths_map):
        raise NotImplementedError

    def load_benchmark_batch(self, sample_paths):
        images = []
        line_embeds = []
        depths = []
        for paths_map in sample_paths:
            line_detector_res = (
                self.load_line_detector_res(paths_map)
                if self.args.do_load_lines
                else None
            )
            sample = self.prep_train_sample(
                paths_map, do_augment=False, line_detector_res=line_detector_res
            )

            sample["image"] = test_transform(sample["image"])

            images.append(sample["image"])
            if self.args.line_op == "concat_embed":
                line_embeds.append(sample["line_embed"])
            depths.append(torch.from_numpy(sample["depth"]))

        res = {
            "image": torch.stack(images),
            "depth": torch.stack(depths),
        }
        if self.args.line_op == "concat_embed":
            res["line_embed"] = torch.stack(line_embeds)
        return res

    def load_intrinsics(self, image_path):
        date_pattern = r"\d{4}_\d{2}_\d{2}"
        date = re.search(date_pattern, image_path).group(0)
        intrinsics = self.intrinsics[date]
        return intrinsics

    def prep_train_sample(self, paths_map, do_augment=False, line_detector_res=None):
        image_path, depth_path = self.load_paths(paths_map)

        adj_imgs_paths = get_adj_img_paths(image_path)
        # adj_imgs = [self.load_rgb(path) for path in adj_imgs_paths]
        adj_imgs = []
        for path in adj_imgs_paths:
            img = self.load_rgb(path)
            img = np.asarray(img, dtype=np.float32) / 255.0
            img = self.prep_img(img)
            adj_imgs.append(img)
        adj_imgs = torch.from_numpy(np.stack(adj_imgs, axis=0)).permute(0, 3, 1, 2)

        image, depth_gt = self.load_img_and_depth_from_path(image_path, depth_path)
        image = np.asarray(image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32)

        include_lines = getattr(self.args, "include_lines", False)
        if include_lines:
            if line_detector_res is None:
                line_detector_res = self.run_line_detector(image)
            lines = line_detector_res["lines"][0].astype(np.int32)
            if self.args.line_filter is not None:
                lines = self.filter_lines(line_detector_res, lines)
        if self.args.line_op == "overlay":
            image = self.overlay_lines(image)
        elif self.args.line_op in ["concat", "concat_binary"]:
            image = self.concat_lines(image, line_detector_res=line_detector_res)
        elif self.args.line_op in ["concat_embed"]:
            line_info = self.get_line_info(image, line_detector_res)
            line_embed = line_info["feature_map"]
            if self.args.do_crop:
                line_embed = kb_crop(line_embed.numpy())

        if self.args.do_crop:
            depth_gt = self.crop(depth_gt)

        orig_shape = image.shape
        image = self.prep_img(image)

        scale_factor_x = self.target_shape[0] / orig_shape[1]
        scale_factor_y = self.target_shape[1] / orig_shape[0]
        intrinsics = self.load_intrinsics(image_path)
        intrinsics[0] *= scale_factor_x
        intrinsics[1] *= scale_factor_y

        depth_gt = resize_inputs(depth_gt, target_shape=self.target_shape)

        depth_gt = np.expand_dims(depth_gt, axis=2)

        if do_augment:
            image, depth_gt = train_preprocess(image, depth_gt)

        depth_gt = self.convert_depth_to_meters(depth_gt)
        depth_gt = np.clip(depth_gt, 1e-3, self.max_depth)

        depth_gt = depth_gt.transpose(2, 0, 1)

        sample = {
            "image": image,
            "depth": depth_gt,
            "adj_imgs": adj_imgs,
            "intrinsics": intrinsics,
        }

        if self.args.line_op in ["concat_embed"]:
            sample["line_embed"] = (
                torch.from_numpy(line_embed)
                .resize_(*image.shape[:2], self.args.line_embed_channels)
                .permute(2, 0, 1)
            )
        if include_lines:
            rescaled_lines = rescale_lines(lines, orig_shape, self.target_shape)
            sample["lines"] = rescaled_lines
        return sample

    def prep_img(self, image):
        if self.args.do_crop:
            image = self.crop(image)

        image = resize_inputs(image, target_shape=self.target_shape)
        if self.args.use_grayscale_img:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=2)
            if image.shape[2] > 3:
                image = np.stack([img, image[..., 3:]], axis=2).squeeze()
            else:
                image = img
        return image

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

    def get_line_info(self, image, line_detector_res=None):
        if line_detector_res is None:
            line_detector_res = self.run_line_detector(image)

        lines = line_detector_res["lines"][0].astype(np.int32)
        if self.args.line_filter is not None:
            lines = self.filter_lines(line_detector_res, lines)

        norm_line_embedding = torch.nn.init.orthogonal_(
            torch.empty(len(lines), self.args.line_embed_channels)
        )
        fm_size = image.shape[:2]
        max_line_x, max_line_y = (
            line_detector_res["lines"][0].max(0).max(0).astype(int) + 1
        )
        fm_size = (max(image.shape[0], max_line_y), max(image.shape[1], max_line_x))
        feature_map = torch.zeros((*fm_size, self.args.line_embed_channels))

        for i, line in enumerate(lines):
            y, x = skimage.draw.line(*(line.astype("int").flatten()))
            feature_map[x, y] = norm_line_embedding[i]
        return {"feature_map": feature_map, "lines": lines}

    def concat_lines(self, image, line_detector_res=None):
        if line_detector_res is None:
            line_detector_res = self.run_line_detector(image)

        lines = self.get_line_mask(image, line_detector_res)
        lines = np.concatenate((image, lines), axis=2)
        return lines

    def get_line_mask(self, image, out):
        if self.args.line_op == "concat":
            df_norm = out["df_norm"][0].cpu().numpy()

            concat = (df_norm - np.min(df_norm)) / (np.max(df_norm) - np.min(df_norm))
            concat = np.expand_dims(concat, axis=2)
        elif self.args.line_op == "concat_binary":
            lines = out["lines"][0].astype(np.int32)

            if self.args.line_filter is not None:
                lines = self.filter_lines(out, lines)

            concat = np.zeros((image.shape[0], image.shape[1], 1))
            for line in lines:
                concat = cv2.line(concat, tuple(line[0]), tuple(line[1]), (1, 1, 1), 2)
        else:
            raise NotImplementedError

        concat = concat.astype(np.float32)
        return concat

    def filter_lines(self, out, lines):
        if "vanishing_point" in self.args.line_filter:
            lines = self.filter_lines_by_vp(lines, out["vp_labels"][0])
        if "length" in self.args.line_filter:
            lines = self.filter_lines_by_length(
                lines,
                min_length=self.args.min_length,
                use_min_length=self.args.use_min_length,
            )
        if "angle" in self.args.line_filter:
            lines = self.filter_lines_by_angle(lines)
        return lines

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

    def filter_lines_by_angle(
        self, lines, low_thresh=np.pi / 10, high_thresh=np.pi / 2.5
    ):
        line_slopes = np.abs(
            (lines[:, 1, 1] - lines[:, 0, 1]) / (lines[:, 1, 0] - lines[:, 0, 0] + 1e-6)
        )
        line_angles = np.arctan(line_slopes)
        new_lines = []

        for idx, line in enumerate(lines):
            if low_thresh < line_angles[idx] < high_thresh:
                new_lines.append(line)
        return np.array(new_lines)

    def filter_lines_by_dist(self, lines):
        # not useful. object contours are naturally close to each other. but! need to look for geometrically identical lines (eg, parallel)
        distances = []
        dist_lines = lines.astype("float32")
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = dist_lines[i]
                line2 = dist_lines[j]
                distance = cv2.pointPolygonTest(line1, (line2[0][0], line2[0][1]), True)
                distances.append((distance, i, j))
        lines_to_rm = []
        dist_thresh = 10
        for distance, i, j in distances:
            if distance < dist_thresh and i not in lines_to_rm:
                lines_to_rm.append(i)
                continue
        lines_filtered = np.delete(lines, lines_to_rm, axis=0)
        return lines_filtered

    def load_img_and_depth(self, paths_map):
        raise NotImplementedError

    def load_img_and_depth_from_path(self, image_path, depth_path):
        raise NotImplementedError

    def load_paths(self, paths_map):
        raise NotImplementedError

    def convert_depth_to_meters(self, depth_gt):
        raise NotImplementedError

    def load_rgb(self, path):
        raise NotImplementedError

    def crop(self, x):
        raise NotImplementedError

    def __len__(self):
        return len(self.filenames)


class KITTIDataset(MonodepthDataset):
    max_depth = 80.0

    def __init__(self, *args_, **kwargs):
        super().__init__(*args_, **kwargs)

        with open(self.args.filenames_file) as json_file:
            json_data = json.load(json_file)
            if self.split == "eigen":
                self.filenames = json_data["eigen"][self.mode]
            else:
                self.filenames = json_data[self.mode]

    def load_paths(self, paths_map):
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

        return image_path, depth_path

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

        return image, depth_gt

    def crop(self, x):
        return kb_crop(x)

    def convert_depth_to_meters(self, depth_gt):
        depth_gt = depth_gt / 256.0
        return depth_gt

    def load_rgb(self, path):
        return Image.open(path)

    def load_line_detector_res(self, paths_map):
        img_path = Path(self.data_dir) / "data_lines" / paths_map["rgb"]
        mask_path = img_path.parent / f"{img_path.stem}.npy"
        return np.load(mask_path, allow_pickle=True).item()
