import numpy as np


def log_batch(
    batch,
    step,
    batch_size,
    experiment,
    prefix=None,
    max_depth=80.0,
):
    for k, v in batch.items():
        log_image_comet(step, batch_size, experiment, prefix, k, v)


def log_image_comet(step, batch_size, experiment, prefix, k, v):
    if len(v.shape) == 3:
        v = v.unsqueeze(0)
    v = v.cpu().numpy().transpose(0, 2, 3, 1)
    v = rescale_img_to_zero_one_range(v)
    for idx in range(batch_size):
        name = f"{k}_{idx}"
        if prefix is not None:
            name = f"{prefix}/{name}"
        experiment.log_image(
            v[idx],
            name,
            step=step,
        )


def optional_normalize_img(x, scaler=255.0):
    if np.max(x) > 1:
        x = x / scaler
    return x


def rescale_img_to_zero_one_range(x):
    return x / np.max(x)
