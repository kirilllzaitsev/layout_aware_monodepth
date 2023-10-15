from pathlib import Path

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
        self.interim_feature_maps = []
        base = self.backbone(x)
        outputs = {}
        # DF embedding
        outputs["df_norm"] = self.df_head(base).squeeze(1)
        # Closest line direction embedding
        outputs["line_level"] = self.angle_head(base).squeeze(1)
        if self.return_both:
            outputs["df_embed"] = self.interim_feature_maps[0]
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
    dlsd = dlsd.eval()
    return dlsd


def load_deeplsd(device, conf=deeplsd_conf):
    from deeplsd.geometry.viz_2d import plot_images, plot_lines
    from deeplsd.models.deeplsd_inference import DeepLSD
    from deeplsd.utils.tensor import batch_to_device

    from layout_aware_monodepth.cfg import cfg

    # Load the model
    if cfg.is_cluster:
        ckpt = "/cluster/work/rsl/kzaitsev/depth_estimation/third_party/DeepLSD/weights/deeplsd_md.tar"
    else:
        ckpt = f"{Path(__file__).parent.parent}/artifacts/deeplsd/deeplsd_md.tar"
    ckpt = torch.load(str(ckpt), map_location="cpu")
    net = DeepLSD(conf)
    net.load_state_dict(ckpt["model"])
    net = net.to(device).eval()
    return net
