import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from self_supervised_dc.dataset_utils.data_logging import logger as logging
from self_supervised_dc.dataset_utils.raw_data_loaders import depth_read, img_read

if not ("DISPLAY" in os.environ):
    import matplotlib as mpl

    mpl.use("Agg")

cmap = plt.cm.jet
cmap2 = plt.cm.nipy_spectral


def validcrop(img):
    ratio = 256 / 1216
    h = img.size()[2]
    w = img.size()[3]
    return img[:, :, h - int(ratio * w) :, :]


def depth_colorize(depth):
    depth = np.squeeze(depth)
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype("uint8")


def feature_colorize(feature):
    feature = (feature - np.min(feature)) / ((np.max(feature) - np.min(feature)))
    feature = 255 * cmap2(feature)[:, :, :3]
    return feature.astype("uint8")


def mask_vis(mask):
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    mask = 255 * mask
    return mask.astype("uint8")


def merge_into_row(
    sample_dict,
    predicted_dm,
):
    """Combines raw inputs and predictions into a single row for visualization.
    The order of the images is: rgb, gray, sparse_dm, predicted_dm, gt_dm
    """

    def preprocess_depth(x):
        y = preprocess_feature(x)
        return depth_colorize(y)

    def preprocess_feature(x):
        if isinstance(x, Image.Image):
            x = np.array(x)
        elif isinstance(x, torch.Tensor):
            x = x.data.cpu().detach().numpy()
        if x.ndim == 4:
            # refers to batched data
            x = x[0, ...]
        y = np.squeeze(x)
        return y

    # if is gray, transforms to rgb
    img_list = []
    if "rgb" in sample_dict:
        rgb = preprocess_feature(sample_dict["rgb"])
        if rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        rgb = resize_img_to_match_another_img_size(predicted_dm, rgb)
        img_list.append(rgb)
    elif "gray" in sample_dict:
        g = preprocess_feature(sample_dict["gray"])
        g = np.array(Image.fromarray(g).convert("RGB"))
        img_list.append(g)
    else:
        raise ValueError("No image found in sample_dict")
    if "sparse_dm" in sample_dict:
        sparse_dm = sample_dict["sparse_dm"]
        sparse_dm = resize_img_to_match_another_img_size(predicted_dm, sparse_dm)
        img_list.append(preprocess_depth(sparse_dm))
        img_list.append(preprocess_depth(predicted_dm))
    if "gt" in sample_dict:
        gt = preprocess_depth(sample_dict["gt"])
        img_list.append(gt)

    img_merge = np.hstack(img_list)
    return img_merge.astype("uint8")


def resize_img_to_match_another_img_size(target, x):
    if x.shape[:2] != target.shape[:2]:
        logging.info(f"Resizing input from {x.shape} to {target.shape}")
        x = cv2.resize(
            x,
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return x


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def display_depth_map(depth, title):
    plt.figure()
    plt.imshow(depth, cmap="jet")
    plt.title(title)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    base_dir = "/media/master/MyPassport/msc_studies/second_semester/research_project/project/layout_aware_monodepth/self_supervised_dc/test_data"
    filename = "0000000035.png"
    sparse_dm = depth_read(f"{base_dir}/lidar/{filename}")
    gt_dm = depth_read(f"{base_dir}/groundtruth/{filename}")
    rgb = img_read(f"{base_dir}/raw_imgs/{filename}")
    # display_depth_map(sparse_dm, "sparse depth map")
    # display_depth_map(gt_dm, "ground truth depth map")
    merged = merge_into_row(
        {
            "rgb": rgb,
            "sparse_dm": sparse_dm,
            "gt": gt_dm,
        },
        predicted_dm=gt_dm,
    )
