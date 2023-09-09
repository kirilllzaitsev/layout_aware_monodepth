import torch
import torch.nn as nn
from deeplsd.models.backbones.vgg_unet import VGGUNet
from deeplsd.models.deeplsd import DeepLSD


class CustomDeepLSD(DeepLSD):
    def __init__(self, conf, return_embedding=True):
        # Base network
        super().__init__(conf)
        self.backbone = VGGUNet(tiny=False)
        dim = 64

        # Predict the distance field and angle to the nearest line
        # DF head
        if not return_embedding:
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
            self.df_head = nn.Sequential(
                *self.get_common_backbone(dim),
            )
            self.angle_head = nn.Sequential(
                *self.get_common_backbone(dim),
            )

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
        base = self.backbone(x)
        outputs = {}
        # DF embedding
        outputs["df_norm"] = self.df_head(base).squeeze(1)
        # Closest line direction embedding
        outputs["line_level"] = self.angle_head(base).squeeze(1)
        return outputs


def load_custom_deeplsd(detect_lines, return_deeplsd_embedding):
    deeplsd_conf = {
        "detect_lines": detect_lines,
        "line_detection_params": {
            "merge": False,
            "filtering": True,
            "grad_thresh": 4,
            "grad_nfa": True,
        },
    }
    ckpt = "../weights/deeplsd/deeplsd_md.tar"
    dlsd = CustomDeepLSD(deeplsd_conf, return_embedding=return_deeplsd_embedding)
    dlsd.load_state_dict(torch.load(str(ckpt))["model"], strict=False)
    return dlsd
