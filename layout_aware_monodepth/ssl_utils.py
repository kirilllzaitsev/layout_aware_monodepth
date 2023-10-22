import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from kbnet import log_utils, losses, net_utils, networks
from rsl_depth_completion.models.benchmarking.calibrated_backprojection_network.kbnet import (
    eval_utils,
)

from layout_aware_monodepth.cfg import cfg


@torch.no_grad()
def compute_triplet_loss(
    image0,
    image1,
    image2,
    output_depth0,
    intrinsics,
    pose01,
    pose02,
    w_color=0.15,
    w_structure=0.95,
    w_sparse_depth=0.60,
    w_smoothness=0.04,
):
    """
    Computes loss function
    l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}

    Arg(s):
        image0 : torch.Tensor[float32]
            N x 3 x H x W image at time step t
        image1 : torch.Tensor[float32]
            N x 3 x H x W image at time step t-1
        image2 : torch.Tensor[float32]
            N x 3 x H x W image at time step t+1
        output_depth0 : torch.Tensor[float32]
            N x 1 x H x W output depth at time t
        sparse_depth0 : torch.Tensor[float32]
            N x 1 x H x W sparse depth at time t
        validity_map_depth0 : torch.Tensor[float32]
            N x 1 x H x W validity map of sparse depth at time t
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics matrix
        pose01 : torch.Tensor[float32]
            N x 4 x 4 relative pose from image at time t to t-1
        pose02 : torch.Tensor[float32]
            N x 4 x 4 relative pose from image at time t to t+1
        w_color : float
            weight of color consistency term
        w_structure : float
            weight of structure consistency term (SSIM)
        w_sparse_depth : float
            weight of sparse depth consistency term
        w_smoothness : float
            weight of local smoothness term
    Returns:
        torch.Tensor[float32] : loss
        dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
    """

    shape = image0.shape
    validity_map_image0 = torch.ones_like(image0)

    # Backproject points to 3D camera coordinates
    points = net_utils.backproject_to_camera(output_depth0, intrinsics, shape)

    # Reproject points onto image 1 and image 2
    target_xy01 = net_utils.project_to_pixel(points, pose01, intrinsics, shape)
    target_xy02 = net_utils.project_to_pixel(points, pose02, intrinsics, shape)

    # Reconstruct image0 from image1 and image2 by reprojection
    image01 = net_utils.grid_sample(image1, target_xy01, shape)
    image02 = net_utils.grid_sample(image2, target_xy02, shape)

    """
    Essential loss terms
    """
    # Color consistency loss function
    loss_color01 = losses.color_consistency_loss_func(
        src=image01, tgt=image0, w=validity_map_image0
    )
    loss_color02 = losses.color_consistency_loss_func(
        src=image02, tgt=image0, w=validity_map_image0
    )
    loss_color = loss_color01 + loss_color02

    # Structural consistency loss function
    loss_structure01 = losses.structural_consistency_loss_func(
        src=image01, tgt=image0, w=validity_map_image0
    )
    loss_structure02 = losses.structural_consistency_loss_func(
        src=image02, tgt=image0, w=validity_map_image0
    )
    loss_structure = loss_structure01 + loss_structure02

    # Local smoothness loss function
    loss_smoothness = losses.smoothness_loss_func(predict=output_depth0, image=image0)

    # l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}
    loss = (
        w_color * loss_color
        + w_structure * loss_structure
        + w_smoothness * loss_smoothness
    )

    loss_info = {
        "loss_color": loss_color,
        "loss_structure": loss_structure,
        "loss_smoothness": loss_smoothness,
        "loss": loss,
        "image01": image01,
        "image02": image02,
    }

    return loss, loss_info


def get_pose_model(device, encoder_type="resnet18"):
    from kbnet.posenet_model import PoseNetModel

    if cfg.is_cluster:
        pose_model_restore_path = "/cluster/home/kzaitse/rsl_depth_completion/rsl_depth_completion/conditional_diffusion/models/kbnet/posenet-kitti.pth"
    else:
        pose_model_restore_path = "/media/master/wext/msc_studies/second_semester/research_project/related_work/calibrated-backprojection-network/pretrained_models/kitti/posenet-kitti.pth"

    pose_model = PoseNetModel(
        encoder_type=encoder_type,
        rotation_parameterization="axis",
        weight_initializer="xavier_normal",
        activation_func="relu",
        device=device,
    )

    pose_model.train()
    pose_model.restore_model(pose_model_restore_path)
    return pose_model


def extract_frame_id_from_img_path(filename: str):
    head, tail = os.path.split(filename)
    number_string = tail[0 : tail.find(".")]
    number = int(number_string)
    return head, number


def get_nearby_img_path(filename: str, new_id: int):
    head, _ = os.path.split(filename)
    new_filename = os.path.join(head, f"{new_id:010d}.png")
    return new_filename


def get_adj_img_paths(path: str, *, n_adjacent=2):
    assert path is not None, "path is None"

    _, frame_id = extract_frame_id_from_img_path(path)
    offsets = []
    for i in range(1, n_adjacent):
        offsets.append(i)
        offsets.append(-i)
    offsets_asc = sorted(offsets)

    adj_imgs_paths = []
    for frame_offset in offsets_asc:
        path_near = get_nearby_img_path(path, frame_id + frame_offset)
        # assert os.path.exists(path_near), f"cannot find two nearby frames for {path}"
        if os.path.exists(path_near):
            adj_imgs_paths.append(path_near)
        else:
            adj_imgs_paths.append(path)
    return adj_imgs_paths
