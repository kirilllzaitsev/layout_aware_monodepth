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
    batch: dict, preds, with_colorbar=False, with_depth_diff=False, max_depth=1
):
    batch_size = len(batch["image"])
    fig, axs = plt.subplots(
        batch_size,
        3 + with_depth_diff,
        figsize=(max(batch_size * 5, 10), (batch_size + with_depth_diff) * 5),
    )
    for i in range(batch_size):
        img = batch["image"][i].permute(1, 2, 0)
        in_depth = batch["depth"][i]
        d = preds[i]

        if batch_size == 1:
            ax_0 = axs[0]
            ax_1 = axs[1]
            ax_2 = axs[2]
            if with_depth_diff:
                ax_3 = axs[3]
        else:
            ax_0 = axs[i, 0]
            ax_1 = axs[i, 1]
            ax_2 = axs[i, 2]
            if with_depth_diff:
                ax_3 = axs[i, 3]
        ax_0.imshow(img)
        ax_1.imshow(in_depth * max_depth)
        ax_2.imshow(d * max_depth)
        axs_row = [ax_0, ax_1, ax_2]
        if with_colorbar:
            attach_colorbar(ax_1, ax_1.images[0], vmax=1 * max_depth)
            attach_colorbar(ax_2, ax_2.images[0], vmax=1 * max_depth)
        if with_depth_diff:
            axs_row.append(ax_3)
        if with_depth_diff:
            diff = (in_depth - d) * max_depth
            ax_3.imshow(diff, cmap="magma")
            attach_colorbar(ax_3, ax_3.images[0], vmin=None, vmax=None)
        for ax in axs_row:
            ax.axis("off")
            ax.set_aspect("equal")
        if i == 0:
            ax_0.set_title("Image", fontsize=20)
            ax_1.set_title("Input Depth", fontsize=20)
            ax_2.set_title("Predicted Depth", fontsize=20)
            if with_depth_diff:
                ax_3.set_title("Depth Difference", fontsize=20)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.0)
    return fig
