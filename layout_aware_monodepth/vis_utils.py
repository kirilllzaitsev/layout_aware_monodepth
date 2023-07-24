import os
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def plot_samples_and_preds(batch: dict, preds):
    batch_size = len(batch["image"])
    fig, axs = plt.subplots(batch_size, 3, figsize=(max(batch_size*5, 10), batch_size*5))
    for i in range(batch_size):
        img = batch["image"][i].permute(1, 2, 0)
        in_depth = batch["depth"][i]
        d = preds[i]

        if batch_size == 1:
            ax_0 = axs[0]
            ax_1 = axs[1]
            ax_2 = axs[2]
        else:
            ax_0 = axs[i, 0]
            ax_1 = axs[i, 1]
            ax_2 = axs[i, 2]
        ax_0.imshow(img)
        ax_1.imshow(in_depth)
        ax_2.imshow(d)
        for ax in [ax_0, ax_1, ax_2]:
            ax.axis("off")
            ax.set_aspect("equal")
        if i == 0:
            ax_0.set_title("Image", fontsize=20)
            ax_1.set_title("Input Depth", fontsize=20)
            ax_2.set_title("Predicted Depth", fontsize=20)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.0)
    return fig
