from pathlib import Path

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
