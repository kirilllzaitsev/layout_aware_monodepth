import os
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image


def attach_colorbar(ax, img=None, vmin=0, vmax=1, scaler=1.0):
    def formatter(x, pos):
        return f"{x * scaler:.0f}"

    img = img or ax.images[0]

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(img, cax=cax, format=formatter)
    cbar.mappable.set_clim(vmin=vmin, vmax=vmax)


def plot_samples_and_preds(
    batch: dict,
    preds,
    with_colorbar=False,
    with_depth_diff=False,
    with_img_depth_overlay=False,
    max_depth=1.0,
    image=None,
    depth=None,
):
    if image is not None:
        batch["image"] = image
    if depth is not None:
        batch["depth"] = depth

    batch_size = len(batch["image"])
    with_lines_concat = batch["image"].shape[1] == 4
    ncols = 3 + with_depth_diff + with_lines_concat + with_img_depth_overlay
    fig, axs = plt.subplots(
        batch_size,
        ncols,
        figsize=(max(batch_size * 5, 10), ncols * 5),
    )
    for i in range(batch_size):
        if with_lines_concat:
            if batch["image"][i].shape[0] == 4:
                img = batch["image"][i][:3]
            else:
                img = batch["image"][i][:1]
            img = img.permute(1, 2, 0)
        else:
            img = batch["image"][i].permute(1, 2, 0)
        in_depth = batch["depth"][i]
        if in_depth.shape[0] == 1:
            in_depth = in_depth.permute(1, 2, 0)
        if in_depth.max() <= 1:
            in_depth *= max_depth
        d = preds[i]
        if d.max() <= 1:
            d *= max_depth
        if d.shape[0] < d.shape[-1]:
            d = d.permute(1, 2, 0)

        if batch_size == 1:
            ax_0 = axs[0]
            ax_1 = axs[1]
            ax_2 = axs[2]
            if with_depth_diff:
                ax_3 = axs[3]
            if with_lines_concat:
                ax_4 = axs[3 + with_depth_diff]
            if with_img_depth_overlay:
                ax_5 = axs[3 + with_depth_diff + with_lines_concat]
        else:
            ax_0 = axs[i, 0]
            ax_1 = axs[i, 1]
            ax_2 = axs[i, 2]
            if with_depth_diff:
                ax_3 = axs[i, 3]
            if with_lines_concat:
                ax_4 = axs[i, 3 + with_depth_diff]
            if with_img_depth_overlay:
                ax_5 = axs[i, 3 + with_depth_diff + with_lines_concat]

        ax_0.imshow(img)
        ax_1.imshow(in_depth)
        ax_2.imshow(d)
        axs_row = [ax_0, ax_1, ax_2]

        if with_colorbar:
            for ax in [ax_1, ax_2]:
                attach_colorbar(ax, ax.images[0], vmax=in_depth.max())

        if with_depth_diff:
            axs_row.append(ax_3)
            diff = np.abs(in_depth - d)
            mask = in_depth > 1e-3
            valid_diff = np.where(mask, diff, 0)
            valid_diff = cv2.dilate(valid_diff, np.ones((2, 2)))
            ax_3.imshow(valid_diff, cmap="magma")
            attach_colorbar(ax_3, ax_3.images[0], vmax=None)

        if with_lines_concat:
            axs_row.append(ax_4)
            ax_4.imshow(batch["image"][i][3])

        if with_img_depth_overlay:
            axs_row.append(ax_5)
            overlay_img_and_depth(ax_5, img, d)

        for ax in axs_row:
            ax.axis("off")
            ax.set_aspect("equal")

        if i == 0:
            fontsize = 16
            ax_0.set_title("Image", fontsize=fontsize)
            ax_1.set_title("Input Depth", fontsize=fontsize)
            ax_2.set_title("Predicted Depth", fontsize=fontsize)
            if with_depth_diff:
                ax_3.set_title("Depth Difference", fontsize=fontsize)
            if with_lines_concat:
                ax_4.set_title("Line channel", fontsize=fontsize)
            if with_img_depth_overlay:
                ax_5.set_title(
                    "Overlaid depth",
                    fontsize=fontsize,
                )
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()
    return fig


def plot_batch(b):
    batch_size = len(b["image"])
    no_line_channel = b["image"].shape[1] == 3
    ncols = 2 if no_line_channel else 3
    fig, axs = plt.subplots(
        batch_size,
        ncols,
        figsize=(max(batch_size * 5, 10), ncols * 5),
    )
    for i, (image, depth) in enumerate(zip(b["image"], b["depth"])):
        if batch_size == 1:
            ax_0 = axs[0]
            ax_1 = axs[1]
            ax_2 = axs[2]
        else:
            ax_0 = axs[i, 0]
            ax_1 = axs[i, 1]
            ax_2 = axs[i, 2]

        if ncols == 2:
            ax_0.imshow(image.permute(1, 2, 0))
        else:
            ax_0.imshow(image[:3].permute(1, 2, 0))
            ax_2.imshow(image[3:].permute(1, 2, 0))
            ax_2.set_title("line channel")
        ax_1.imshow(depth)
        ax_0.set_title("image")
        ax_1.set_title("depth")
    for ax in axs.flatten():
        ax.axis("off")
    plt.show()


def plot_attention(*args, method="rollout", **kwargs):
    if method == "rollout":
        return plot_attention_rollout(*args, **kwargs)
    elif method == "heatmap":
        return plot_attention_heatmap(*args, **kwargs)
    else:
        raise ValueError(f"Unknown attention method {method}")


def plot_attention_heatmap(attention_scores, img_h, img_w):
    num_tokens = 0
    PATCH_SIZE = 4
    num_heads = 4

    # (4, 192, 64)
    # torch.Size([3072, 4, 16, 16])

    # image = torch.randn(1, 3, 256, 256)
    # image = torch.randn(1, 3, 64*2, 192*2)
    # image = torch.randn(1, 3, 256 // 4, 768 // 4)
    # Process the attention maps for overlay.
    w_featmap = img_w // PATCH_SIZE
    h_featmap = img_h // PATCH_SIZE
    # attention_scores = torch.randn(
    #     # 3072, num_heads, PATCH_SIZE, PATCH_SIZE
    #     # 1 * w_featmap * h_featmap, num_heads, PATCH_SIZE, PATCH_SIZE
    #     1 * w_featmap * h_featmap,
    #     num_heads,
    #     PATCH_SIZE**2,
    #     PATCH_SIZE**2,
    # )

    # Sort the Transformer blocks in order of their depth.

    # w_featmap,h_featmap=192*2//4,64*2//4
    # Taking the representations from CLS token.
    # BxNxHxW
    attention_scores = rearrange(
        attention_scores,
        # TODO: the following line is wrong
        "(b n1 n2) c p1 p2 -> b c (n1 p1) (n2 p2)",
        n1=w_featmap,
        n2=h_featmap,
    )
    # BxNxHxW
    attentions = attention_scores[0, :, 0, num_tokens:].reshape(
        num_heads, w_featmap, h_featmap
    )

    # Reshape the attention scores to resemble mini patches.
    attentions = attentions.permute((1, 2, 0))

    # Resize the attention patches to 224x224 (224: 14x16).
    dsize = (img_h, img_w)
    # dsize = (h_featmap * PATCH_SIZE, w_featmap * PATCH_SIZE)
    # attentions = (attentions - attentions.min()) / (attentions.max() - attentions.min())
    attentions = cv2.resize(attentions.numpy(), dsize=dsize)
    return attentions


# attention_scores = torch.randn(3072, 4, 16, 16)
# image = torch.randn(1, 3, 256//4, 768//4)
# x=plot_attention_heatmap(attention_scores, image)
# print(x.shape)
# exit(0)


def plot_attention_rollout(att_mat, img_size, heads_agg="mean"):
    # att_mat = torch.stack(att_mat).squeeze(1)

    if heads_agg == "mean":
        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=1)
    elif heads_agg == "min":
        # Take the maximum across all heads.
        att_mat = torch.min(att_mat, dim=1)[0]
    elif heads_agg == "max":
        # Take the maximum across all heads.
        att_mat = torch.max(att_mat, dim=1)[0]
    elif heads_agg == "last":
        # Take the last head.
        att_mat = att_mat[:, -1]
    else:
        raise ValueError(f"Unknown heads_agg {heads_agg}")

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 0:].reshape(grid_size, grid_size).detach().numpy()
    # mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    import cv2

    print(mask.shape)
    mask = cv2.resize(mask / mask.max(), img_size)[..., np.newaxis]
    return mask


def overlay_img_and_depth(ax, img, depth, depth_alpha=0.6):
    if img.shape[0] < img.shape[-1]:
        img = img.permute(1, 2, 0)
    ax.imshow(img)

    cmap = plt.get_cmap("magma")

    vmin = 0
    vmax = depth.max()

    # Create a new axes for the colorbar adjacent to the main axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Attach colorbar to the new axes
    cbar = plt.colorbar(
        mappable=cm.ScalarMappable(
            cmap=cmap, norm=cm.colors.Normalize(vmin=vmin, vmax=vmax)
        ),
        cax=cax,
    )
    cbar.set_label("Depth")

    ax.imshow(depth, cmap=cmap, alpha=depth_alpha, vmin=vmin, vmax=vmax)
    return ax


def get_pointcloud_from_rgbd(
    image: np.array,
    depth: np.array,
    mask: np.ndarray,
    intrinsic_matrix: np.array,
    extrinsic_matrix: np.array = None,
):
    depth = np.array(depth).squeeze()
    mask = np.array(mask).squeeze()
    # Mask the depth array
    masked_depth = np.ma.masked_where(mask == False, depth)
    # masked_depth = np.ma.masked_greater(masked_depth, 8000)
    # Create idx array
    idxs = np.indices(masked_depth.shape)
    u_idxs = idxs[1]
    v_idxs = idxs[0]
    # Get only non-masked depth and idxs
    z = masked_depth[~masked_depth.mask]
    compressed_u_idxs = u_idxs[~masked_depth.mask]
    compressed_v_idxs = v_idxs[~masked_depth.mask]
    image = np.stack(
        [image[..., i][~masked_depth.mask] for i in range(image.shape[-1])], axis=-1
    )

    # Calculate local position of each point
    # Apply vectorized math to depth using compressed arrays
    cx = intrinsic_matrix[0, 2]
    fx = intrinsic_matrix[0, 0]
    x = (compressed_u_idxs - cx) * z / fx
    cy = intrinsic_matrix[1, 2]
    fy = intrinsic_matrix[1, 1]
    # Flip y as we want +y pointing up not down
    y = -((compressed_v_idxs - cy) * z / fy)

    # # Apply camera_matrix to pointcloud as to get the pointcloud in world coords
    if extrinsic_matrix is not None:
        # Calculate camera pose from extrinsic matrix
        camera_matrix = np.linalg.inv(extrinsic_matrix)
        # Create homogenous array of vectors by adding 4th entry of 1
        # At the same time flip z as for eye space the camera is looking down the -z axis
        w = np.ones(z.shape)
        x_y_z_eye_hom = np.vstack((x, y, -z, w))
        # Transform the points from eye space to world space
        x_y_z_world = np.dot(camera_matrix, x_y_z_eye_hom)[:3]
        return x_y_z_world.T
    else:
        x_y_z_local = np.stack((x, y, z), axis=-1)
    return np.concatenate([x_y_z_local, image], axis=-1)
