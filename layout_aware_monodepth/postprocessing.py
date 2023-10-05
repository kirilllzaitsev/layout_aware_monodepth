"""Following AdaBins postprocessing steps when evaluating results"""

import numpy as np
import torch.nn as nn


def postproc_eval_depths(pred, target, min_depth=1e-3, max_depth=10):
    pred = nn.functional.interpolate(
        pred, target.shape[-2:], mode="bilinear", align_corners=True
    )

    pred = pred.detach().squeeze().cpu().numpy()
    pred[pred < min_depth] = min_depth
    pred[pred > max_depth] = max_depth
    pred[np.isinf(pred)] = max_depth
    pred[np.isnan(pred)] = min_depth

    gt_depth = target.squeeze().cpu().numpy()
    return pred, gt_depth


def compute_eval_mask(
    gt_depth, min_depth, max_depth, ds_name, crop_type=None
):
    valid_mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
    if crop_type is not None:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if crop_type == "garg":
            eval_mask[
                int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
            ] = 1

        elif crop_type == "eigen":
            if ds_name == "kitti":
                eval_mask[
                    int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                ] = 1
            else:
                eval_mask[45:471, 41:601] = 1
        valid_mask = np.logical_and(valid_mask, eval_mask)
    return valid_mask
