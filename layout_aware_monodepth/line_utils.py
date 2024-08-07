import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from deeplsd.models.backbones.vgg_unet import VGGUNet
from deeplsd.models.deeplsd import DeepLSD


class CustomDeepLSD(DeepLSD):
    def __init__(self, conf, return_embedding=True, return_both=False):
        # Base network
        super().__init__(conf)
        self.backbone = VGGUNet(tiny=False)
        self.return_both = return_both
        dim = 64

        # Predict the distance field and angle to the nearest line
        # DF head
        if not return_embedding or return_both:
            self.df_head = nn.Sequential(
                *self.get_common_backbone(dim),
                nn.Conv2d(64, 1, kernel_size=1),
                nn.ReLU(),
            )
            # Closest line direction head
            self.angle_head = nn.Sequential(
                *self.get_common_backbone(dim),
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.df_extractor = nn.Sequential(
                *self.get_common_backbone(dim),
            )
            self.angle_extractor = nn.Sequential(
                *self.get_common_backbone(dim),
            )
            self.df_head = self.df_extractor
            self.angle_head = self.angle_extractor

        self.interim_feature_maps = []

        # Register hooks to capture feature maps
        idx_df_feature_map_before_final_proj = 3
        for layer_idx in [idx_df_feature_map_before_final_proj]:
            self.df_head[layer_idx].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.interim_feature_maps.append(output)

    def get_common_backbone(self, dim):
        return (
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        # x must be dict containing an 'image' key with a normalized image
        if isinstance(x, dict):
            x = x["image"]
        self.interim_feature_maps = []
        base = self.backbone(x)
        outputs = {}
        # DF embedding
        outputs["df_norm"] = self.df_head(base).squeeze(1)
        # Closest line direction embedding
        outputs["line_level"] = self.angle_head(base).squeeze(1)
        if self.return_both:
            outputs["df_embed"] = self.interim_feature_maps[0]

        # # Detect line segments
        # if self.conf.detect_lines:
        #     lines = []
        #     np_img = (x.cpu().numpy()[:, 0] * 255).astype(np.uint8)
        #     outputs["df"] = self.denormalize_df(outputs["df_norm"])
        #     np_df = outputs["df"].cpu().numpy()
        #     np_ll = outputs["line_level"].cpu().numpy() * np.pi
        #     if len(np_df.shape) == 4:
        #         np_df = np_df.squeeze(1)
        #         np_ll = np_ll.squeeze(1)
        #     vps, vp_labels = [], []
        #     for img, df, ll in zip(np_img, np_df, np_ll):
        #         line, label, vp = self.detect_afm_lines(
        #             img, df, ll, **self.conf.line_detection_params
        #         )
        #         lines.append(line)
        #         vp_labels.append(label)
        #         vps.append(vp)
        #     outputs["vp_labels"] = vp_labels
        #     outputs["vps"] = vps
        #     outputs["lines"] = lines

        return outputs


deeplsd_conf = {
    "detect_lines": True,
    "line_detection_params": {
        "merge": False,
        "filtering": True,
        "grad_thresh": 4,
        "grad_nfa": True,
    },
}


def load_custom_deeplsd(detect_lines, return_deeplsd_embedding, return_both=False):
    conf = deeplsd_conf.copy()
    conf["detect_lines"] = detect_lines
    ckpt = f"{Path(__file__).parent.parent}/artifacts/deeplsd/deeplsd_md.tar"
    dlsd = CustomDeepLSD(
        deeplsd_conf, return_embedding=return_deeplsd_embedding, return_both=return_both
    )
    dlsd.load_state_dict(torch.load(str(ckpt))["model"], strict=False)
    for p in dlsd.parameters():
        p.requires_grad = False
    dlsd = dlsd.eval()
    return dlsd


def load_deeplsd(conf=deeplsd_conf):
    from deeplsd.models.deeplsd_inference import DeepLSD

    # Load the model
    ckpt = f"{Path(__file__).parent.parent}/artifacts/deeplsd/deeplsd_md.tar"
    ckpt = torch.load(str(ckpt), map_location="cpu")
    net = DeepLSD(conf)
    net.load_state_dict(ckpt["model"])
    for p in net.parameters():
        p.requires_grad = False
    net = net.eval()
    return net


def get_deeplsd_pred(deeplsd, x):
    gray_img = x.mean(dim=1, keepdim=True)
    line_res = deeplsd({"image": gray_img})
    return line_res


def filter_nearby_vps(vps_mapped):
    vps_to_plot = []
    vps_to_rm = []
    for i in range(len(vps_mapped)):
        for j in range(i + 1, len(vps_mapped)):
            dist = np.linalg.norm(vps_mapped[i] - vps_mapped[j])
            if dist < 50:
                vps_to_plot.append(vps_mapped[i])
                vps_to_plot.append(vps_mapped[j])
                vps_to_rm.append(j)
    vps_mapped = [vp for i, vp in enumerate(vps_mapped) if i not in vps_to_rm]
    return vps_mapped


def proj_vps_to_img(line_res):
    vps = line_res["vps"][0]
    vps_mapped = [
        np.array([vp[0] / vp[2], vp[1] / vp[2]]).astype(np.int32) for vp in vps
    ]
    return vps_mapped


def rescale_lines(lines, orig_shape, target_shape):
    return lines * np.array(
        [
            target_shape[0] / orig_shape[1],
            target_shape[1] / orig_shape[0],
        ]
    )


def filter_lines_by_length(batched_lines, min_length=10, use_min_length=False):
    if not isinstance(batched_lines, list):
        batched_lines = [batched_lines]

    filtered_lines = []
    for lines in batched_lines:
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
        filtered_lines.append(np.array(new_lines))
    return filtered_lines if len(filtered_lines) > 1 else filtered_lines[0]


def filter_lines_by_angle(
    batched_lines, low_thresh=np.pi / 10, high_thresh=np.pi / 2.5
):
    if not isinstance(batched_lines, list):
        batched_lines = [batched_lines]

    filtered_lines = []
    for lines in batched_lines:
        line_angles_rad = get_line_angles(lines)
        new_lines = []

        for idx, line in enumerate(lines):
            if low_thresh < line_angles_rad[idx] < high_thresh:
                new_lines.append(line)
        filtered_lines.append(np.array(new_lines))
    return filtered_lines if len(filtered_lines) > 1 else filtered_lines[0]


def get_line_angles(lines):
    line_slopes = torch.abs(
        (lines[:, 1, 1] - lines[:, 0, 1]) / (lines[:, 1, 0] - lines[:, 0, 0] + 1e-6)
    )
    line_angles_rad = torch.arctan(line_slopes)
    return line_angles_rad


def line_distance_loss(avg_distances):
    return torch.mean(avg_distances)


def line_orientation_loss(lines1, lines2):
    if not isinstance(lines1, torch.Tensor):
        lines1 = torch.from_numpy(lines1)
    if not isinstance(lines2, torch.Tensor):
        lines2 = torch.from_numpy(lines2)
    angles1 = get_line_angles(lines1)
    angles2 = get_line_angles(lines2)
    return torch.mean(torch.abs(angles1 - angles2))


def reproject_lines(lines, pose, K, Ki, depth):
    line_coords = torch.from_numpy(lines).to(pose.device)
    line_coords[:, :, 1] = torch.clamp(line_coords[:, :, 1], 0, depth.shape[-2] - 1)
    line_coords[:, :, 0] = torch.clamp(line_coords[:, :, 0], 0, depth.shape[-1] - 1)
    line_coords = torch.cat(
        [line_coords, torch.ones(len(line_coords), 1, 2).to(line_coords.device)], 1
    )
    line_depth = depth[0, line_coords.long()[:, :2, 1], line_coords.long()[:, :2, 0]]
    cam_points = torch.matmul(Ki[..., :3, :3].float(), line_coords)
    cam_points = line_depth.view(len(line_depth), 1, -1) * cam_points
    world_points = torch.matmul(
        pose[None, :3, :3].transpose(1, 2), cam_points - pose[None, :3, 3].unsqueeze(-1)
    )

    P = torch.matmul(K, pose)[None, :3, :]
    reproj_cam_points = torch.matmul(
        P,
        torch.cat(
            [world_points, torch.ones(len(line_coords), 1, 2).to(line_coords.device)], 1
        ),
    )

    reproj_line_coords = reproj_cam_points[:, :2, :] / (
        reproj_cam_points[:, 2, :].unsqueeze(1) + 1e-8
    )
    reproj_line_coords = reproj_line_coords.detach().int()
    return reproj_line_coords


def reproject_lines_batch(lines, pose, K, Ki, depth):
    res = []
    if len(K.shape) == 2:
        K = K[None]
        Ki = Ki[None]
    for lines_, pose_, K_, Ki_, depth_ in zip(lines, pose, K, Ki, depth):
        res.append(reproject_lines(lines_, pose_, K_, Ki_, depth_))
    return res


def find_closest_lines_to_src(src, target):
    if not isinstance(src, torch.Tensor):
        src = torch.from_numpy(src)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target).to(src.device)
    dist_start = torch.linalg.norm(src[:, None, 0, :] - target[None, :, 0, :], dim=2)
    dist_end = torch.linalg.norm(src[:, None, 1, :] - target[None, :, 1, :], dim=2)

    # Average the distances from start and end points
    avg_distances = (dist_start + dist_end) / 2

    # Find the index of the closest line for each reprojected line
    closest_line_avg_distances, closest_line_indices = torch.min(avg_distances, dim=1)

    if len(src) != len(target):
        # remove lines that mapped to the same target line
        new_closest_idxs = []
        new_reproj_idxs = []
        for idx_reproj, idx_closest in enumerate(closest_line_indices):
            if idx_closest not in new_closest_idxs:
                new_closest_idxs.append(idx_closest)
                new_reproj_idxs.append(idx_reproj)
        closest_line_indices = torch.tensor(new_closest_idxs)
        src = src[new_reproj_idxs]
        closest_line_avg_distances = closest_line_avg_distances[new_reproj_idxs]

    # Select the closest lines
    closest_lines = target[closest_line_indices].int()
    return {
        "paired_lines": {
            "reproj": src,
            "true": closest_lines,
        },
        "avg_distances": closest_line_avg_distances,
    }


def find_closest_lines_to_src_batch(src, target):
    res = []
    for src_, target_ in zip(src, target):
        res.append(find_closest_lines_to_src(src_, target_))
    return res


def plot_line_pairs(
    reprojected, true, hw, take_n=None, avg_distances=None, font_scale=1, no_text=True
):
    # canvas = np.zeros(hw)
    canvas = np.zeros((hw[0], hw[1], 3))
    if take_n is not None:
        if avg_distances is not None:
            worst_idxs = torch.argsort(avg_distances, descending=True)
            reprojected = reprojected[worst_idxs]
            true = true[worst_idxs]
        reprojected = reprojected[:take_n]
        true = true[:take_n]
    # colors = (
    #     (np.linspace(0.2, 1, reprojected.shape[0])),
    #     (np.linspace(0.2, 1, reprojected.shape[0])),
    #     (np.linspace(0.2, 1, reprojected.shape[0])),
    # )
    colors = (
        np.random.rand(reprojected.shape[0]),
        np.random.rand(reprojected.shape[0]),
        np.random.rand(reprojected.shape[0]),
    )
    if not isinstance(reprojected, np.ndarray):
        reprojected = reprojected.cpu().numpy()
    if not isinstance(true, np.ndarray):
        true = true.cpu().numpy()
    for i in range(reprojected.shape[0]):
        pair_color = colors[0][i], colors[1][i], colors[2][i]
        canvas = cv2.line(
            canvas,
            tuple(reprojected[i, 0]),
            tuple(reprojected[i, 1]),
            pair_color,
            2,
        )
        canvas = cv2.line(
            canvas,
            tuple(true[i, 0]),
            tuple(true[i, 1]),
            pair_color,
            2,
        )
        if not no_text:
            # mark line in each pair with a single index
            canvas = cv2.putText(
                canvas,
                str(i),
                tuple(reprojected[i, 0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4 * font_scale,
                (1, 1, 1),
                2,
                cv2.LINE_AA,
            )
            canvas = cv2.putText(
                canvas,
                str(i),
                tuple(true[i, 0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4 * font_scale,
                (1, 1, 1),
                2,
                cv2.LINE_AA,
            )
    return canvas


def plot_lines(lines, hw, color=(1, 1, 1)):
    if not isinstance(lines, np.ndarray):
        lines = lines.cpu().numpy()
    lines = lines.astype("int")
    # overlay = image.permute(1, 2, 0).numpy().copy()
    overlay = np.zeros(hw)
    for line in lines:
        overlay = cv2.line(overlay, tuple(line[0]), tuple(line[1]), color)
    return overlay


import skimage


def line_eq_parametric(t, line_p1, line_vec_norm):
    return line_p1 + t.reshape(-1, 1) * line_vec_norm


def np_close(np_val, a=1e-3):
    return np.isclose(np_val, a, atol=1e-4)


def infill_depth_along_line_3d(full_gt, lines, K):
    if not isinstance(K, torch.Tensor):
        K = torch.from_numpy(K).float()
    try:
        inv_K = torch.inverse(K).unsqueeze(0)
    except np.linalg.LinAlgError as e:
        print(f"K is not invertible: {K}", str(e))
        return full_gt
    do_unsqueeze = False
    if len(full_gt.shape) == 3:
        full_gt = full_gt.squeeze()
        do_unsqueeze = True

    lines[:, :, 1] = np.clip(lines[:, :, 1], 0, full_gt.shape[-2] - 1)
    lines[:, :, 0] = np.clip(lines[:, :, 0], 0, full_gt.shape[-1] - 1)

    new_gt = full_gt.copy()
    counter = 0
    backproj_lines = []
    added_pts = []
    for i, line in enumerate(lines):
        # coords of pts on a line
        y, x = skimage.draw.line(*(line.astype("int").flatten()))
        # depth of pts on a line
        line_depths = torch.from_numpy(full_gt)[x, y]
        # if not enough depth points, only two are enough to form a line, skip
        if len(line_depths[line_depths > 1e-3]) < 2:
            # print("no depth")
            continue
        # enough points, consider line for backprojection
        counter += 1

        # mask for zero-depth points
        zero_depth_mask = line_depths <= 1e-3
        # y = y[zero_depth_mask]
        # x = x[zero_depth_mask]

        # 3d projection start
        pix_coords = torch.unsqueeze(
            torch.stack([torch.from_numpy(x), torch.from_numpy(y)], 0), 0
        )
        pix_coords = pix_coords.repeat(1, 1, 1)
        pix_coords = (
            torch.cat([pix_coords, torch.ones_like(pix_coords[:, :1])], 1)
        ).float()
        cam_points = torch.matmul(inv_K[:, :3, :3], pix_coords)
        # multiply by depth. many of the points will be zero-depth
        cam_points = line_depths.view(1, -1) * cam_points
        cam_points = torch.cat([cam_points, torch.ones_like(cam_points[:, :1])], 1)
        # 3d projection end

        cam_points = cam_points[:, :, cam_points[0, 2] > 1e-3]
        # cam_points are sorted by depth in ascending order
        # take that start and end points of the line
        line_p1 = cam_points[..., 0]
        line_p2 = cam_points[..., -1]
        # form a vector from start to end
        line_vec = line_p2 - line_p1
        # normalize the vector to get the direction
        line_length = torch.linalg.norm(line_vec)
        line_vec_norm = line_vec / line_length

        # sampling of pts should be bounded by the length of the line to avoid points in unknown space beyond the line segment
        scaler = line_length
        t_vals = torch.linspace(0, 1 * scaler, len(y))
        # uniformly sample points on a 3d line. their z values already have depth and are meaningful
        backproj_line_pts = line_eq_parametric(t_vals, line_p1, line_vec_norm)
        backproj_line_pts = backproj_line_pts.T.unsqueeze(0)

        # vis_line_with_pts_on_it_3d(
        #     torch.vstack([line_p1, line_p2])[:,:-1].unsqueeze(0).numpy().reshape(-1, 3),
        #     backproj_line_pts.squeeze().numpy()[:3].T
        # )

        # get the 2d projection of the 3d points
        # P = torch.matmul(K, pose)[..., :3, :]
        # since we don't use pose, P is just K
        P = K.unsqueeze(0)

        # cam_points = torch.matmul(P, cam_points)
        # projection of 3d points to 2d
        cam_points_reproj = torch.matmul(P, backproj_line_pts)
        # cam_points_reproj = backproj_line_pts

        # normalize by z
        pix_coords_line = cam_points_reproj[:, :2, :] / (
            cam_points_reproj[:, 2, :].unsqueeze(1) + 1e-7
        )
        # come back to line coords
        pix_coords_line = pix_coords_line.view(2, cam_points_reproj.shape[-1])
        # why is the scaling here? does it hold for the lines instead of image?
        # pix_coords_line[..., 0] /= args.width - 1
        # pix_coords_line[..., 1] /= args.height - 1
        # pix_coords_line = (pix_coords_line - 0.5) * 2

        # clip projected coords to image size
        pix_coords_line[0, pix_coords_line[0] >= full_gt.shape[0]] = (
            full_gt.shape[0] - 1
        )
        pix_coords_line[1, pix_coords_line[1] >= full_gt.shape[1]] = (
            full_gt.shape[1] - 1
        )
        # filter out points that are outside of the image (although can clamp as well, but have to be careful with huge values when clamping)
        valid_coords = torch.logical_and(
            pix_coords_line[0] >= 0, pix_coords_line[1] >= 0
        )
        pix_coords_line = pix_coords_line[:, valid_coords].long().numpy()

        # if pixels are non-zero on gt depth, skip adding them
        pix_coords_line_with_missing_gt = []
        idxs_of_coords_with_missing_gt = []
        for i, coords in enumerate(pix_coords_line.T):
            coord_x, coord_y = coords
            if not np.isclose(full_gt[coord_x, coord_y], 1e-3):
                continue
            pix_coords_line_with_missing_gt.append(coords)
            idxs_of_coords_with_missing_gt.append(i)

        # should not happen
        if len(pix_coords_line_with_missing_gt) == 0:
            print("no missing gt at all reprojected coords. skipping line")
            added_pts.append(0)
            continue
        pix_coords_line_with_missing_gt = np.array(pix_coords_line_with_missing_gt).T

        # depth is the third coordinate of pts in 3d space
        backproj_line_pts_depth = backproj_line_pts[:, 2]
        # add the depth to the gt at coords where gt was zero
        new_gt[
            pix_coords_line_with_missing_gt[0], pix_coords_line_with_missing_gt[1]
        ] = backproj_line_pts_depth.squeeze()[valid_coords][
            idxs_of_coords_with_missing_gt
        ]

        backproj_lines.append(pix_coords_line_with_missing_gt)
        added_pts.append(len(pix_coords_line_with_missing_gt[0]))
    # print(f"{counter=}, {len(lines)=}, len(lines)/#backproj={len(lines)/counter}")
    # print(f"{np.mean(added_pts)=}")
    # print(np.sum(full_gt > 1e-3), np.sum(new_gt > 1e-3))
    # print(f"#nonzero_old/#nonzero_new{np.sum(full_gt > 1e-3)/np.sum(new_gt > 1e-3)=}")
    if do_unsqueeze:
        new_gt = new_gt[None]
    return {"infilled_gt": new_gt, "backproj_lines": backproj_lines}
