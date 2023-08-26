import os
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image


def attach_colorbar(ax, img, vmin=0, vmax=1):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(img, cax=cax)
    cbar.mappable.set_clim(vmin=vmin, vmax=vmax)


def plot_samples_and_preds(
    batch: dict, preds, with_colorbar=False, with_depth_diff=False, max_depth=1.0
):
    batch_size = len(batch["image"])
    with_lines_concat = batch["image"].shape[1] == 4
    ncols = 3 + with_depth_diff + with_lines_concat
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
        in_depth = batch["depth"][i] * max_depth
        d = preds[i] * max_depth

        if batch_size == 1:
            ax_0 = axs[0]
            ax_1 = axs[1]
            ax_2 = axs[2]
            if with_depth_diff:
                ax_3 = axs[3]
            if with_lines_concat:
                ax_4 = axs[4]
        else:
            ax_0 = axs[i, 0]
            ax_1 = axs[i, 1]
            ax_2 = axs[i, 2]
            if with_depth_diff:
                ax_3 = axs[i, 3]
            if with_lines_concat:
                ax_4 = axs[i, 4]

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
            ax_3.imshow(diff, cmap="magma")
            attach_colorbar(ax_3, ax_3.images[0], vmax=None)

        if with_lines_concat:
            axs_row.append(ax_4)
            ax_4.imshow(batch["image"][i][3])

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
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()
    return fig


def plot_batch(b):
    for image, depth in zip(b["image"], b["depth"]):
        ncols = 2 if image.shape[0] == 3 else 3
        fig, axs = plt.subplots(1, ncols)
        if ncols == 2:
            axs[0].imshow(image.permute(1, 2, 0))
        else:
            axs[0].imshow(image[:3].permute(1, 2, 0))
            axs[2].imshow(image[3:].permute(1, 2, 0))
            axs[2].set_title("line channel")
        axs[1].imshow(depth)
        axs[0].set_title("image")
        axs[1].set_title("depth")
        for ax in axs[1:]:
            ax.axis("off")
        plt.show()


def plot_attention(att_mat, img_size):
    # att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

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
