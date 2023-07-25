import os

import numpy as np
from PIL import Image

from layout_aware_monodepth.logger import logger as logging


def save_results(predictions, predict_dataloader, output_path):
    logging.info(f"Saving outputs to {output_path}")

    image_dirpath = os.path.join(output_path, "image")
    output_depth_dirpath = os.path.join(output_path, "output_depth")
    sparse_depth_dirpath = os.path.join(output_path, "sparse_depth")
    ground_truth_dirpath = os.path.join(output_path, "ground_truth")

    dirpaths = [
        image_dirpath,
        output_depth_dirpath,
        sparse_depth_dirpath,
        ground_truth_dirpath,
    ]

    for dirpath in dirpaths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    batch_size = predict_dataloader.batch_size
    for batch_idx, ((images, sparse_depths, _), output_depths) in enumerate(
        zip(predict_dataloader, predictions)
    ):
        for local_idx, (image, sparse_depth, output_depth) in enumerate(
            zip(images, sparse_depths, output_depths)
        ):
            filename = "{:010d}.png".format(batch_idx * batch_size + local_idx)
            image = image.cpu().numpy().squeeze().transpose(1, 2, 0)
            sparse_depth = sparse_depth.cpu().numpy().squeeze()
            output_depth = output_depth.detach().cpu().numpy().squeeze()
            save_sample(
                image_dirpath,
                output_depth_dirpath,
                sparse_depth_dirpath,
                image,
                sparse_depth,
                output_depth,
                filename,
            )


def save_sample(
    image_dirpath,
    output_depth_dirpath,
    sparse_depth_dirpath,
    image,
    sparse_depth,
    output_depth,
    filename,
):
    image_path = os.path.join(image_dirpath, filename)
    image = (255 * image).astype(np.uint8)
    Image.fromarray(image).save(image_path)

    output_depth_path = os.path.join(output_depth_dirpath, filename)
    save_depth(output_depth, output_depth_path)

    sparse_depth_path = os.path.join(sparse_depth_dirpath, filename)
    save_depth(sparse_depth, sparse_depth_path)


def save_depth(z, path):
    """
    Saves a depth map to a 16-bit PNG file

    Arg(s):
        z : numpy[float32]
            depth map
        path : str
            path to store depth map
    """

    z = np.uint32(z * 256.0)
    z = Image.fromarray(z, mode="I")
    z.save(path)
