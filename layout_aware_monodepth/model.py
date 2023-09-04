import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from alternet.models.alternet import AttentionBasicBlockB
from deeplsd.models.backbones.vgg_unet import VGGUNet
from deeplsd.models.deeplsd import DeepLSD
from einops import einsum, rearrange

encoder_to_last_channels_in_level = {
    "timm-mobilenetv3_large_100": [16, 24, 40, 80, 112, 160, 960],
    "resnet18": [64, 128, 256, 512],
}
skip_conn_channels = {
    "timm-mobilenetv3_large_100": [3, 16, 24, 40, 112, 960],
    "resnet18": [16, 64, 64, 128, 256, 512],
}


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, heads, head_channel, dropout=0.0):
        super().__init__()
        inner_dim = heads * head_channel
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = head_channel**-0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, head_channel), nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, c, h, w = x.shape
        # q =  x.reshape(b, c, h*w).transpose(1,2).unsqueeze(1)
        # k = self.to_k(z).view(b, self.heads, m, c)
        # v = self.to_v(z).view(b, self.heads, m, c)
        # dots = q @ k.transpose(2, 3) * self.scale
        z = rearrange(z, "b c h w -> b (h w) c")
        q = rearrange(x, "b c h w -> b (h w) c").unsqueeze(1)
        k = rearrange(self.to_k(z), "b m (h c) -> b h m c", h=self.heads)
        v = rearrange(self.to_v(z), "b m (h c) -> b h m c", h=self.heads)
        dots = einsum(q, k, "b h l c, b h m c -> b h l m") * self.scale
        attn = self.attend(dots)
        out = attn @ v
        out = rearrange(out, "b h l c -> b l (h c)")
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, heads, head_channel, dropout=0.0):
        super().__init__()
        inner_dim = heads * head_channel
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.scale = head_channel**-0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, head_channel), nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        q, k, v = torch.chunk(self.to_qkv(x), 3, dim=-1)
        dots = einsum(q, k, "b h l c, b h m c -> b h l m") * self.scale
        attn = self.attend(dots)
        out = attn @ v
        out = rearrange(out, "b h l c -> b l (h c)")
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out


class CustomDeepLSD(DeepLSD):
    def __init__(self, conf):
        # Base network
        super().__init__(conf)
        self.backbone = VGGUNet(tiny=False)
        dim = 64

        # Predict the distance field and angle to the nearest line
        # DF head
        self.df_head = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # Closest line direction head
        self.angle_head = nn.Sequential(
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


class DepthModel(nn.Module):
    # initializers
    def __init__(
        self,
        encoder_name="timm-mobilenetv3_large_100",
        encoder_weights="imagenet",
        decoder_activation="sigmoid",
        decoder_attention_type=None,
        decoder_first_channel=256,
        window_size=4,
        do_insert_after=True,
        in_channels=3,
        use_attn=False,
        use_extra_conv=False,
        do_attend_line_info=False,
    ):
        super().__init__()

        decoder_channels = [decoder_first_channel]
        for i in range(1, 5):
            decoder_channels.append(decoder_first_channel // (2**i))
        self.encoder_name = encoder_name
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=decoder_activation,
            encoder_depth=len(decoder_channels),
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
        )

        self.encoder = model.encoder
        deeplsd_conf = {
            "detect_lines": True,  # Whether to detect lines or only DF/AF
            "line_detection_params": {
                "merge": False,
                "filtering": True,
                "grad_thresh": 4,
                "grad_nfa": True,
            },
        }
        ckpt = "../weights/deeplsd/deeplsd_md.tar"
        self.dlsd = CustomDeepLSD(deeplsd_conf)
        self.dlsd.load_state_dict(
            torch.load(str(ckpt), map_location="cpu")["model"], strict=False
        )

        self.attend_line_info = do_attend_line_info
        self.line_attn_blocks = []
        if self.attend_line_info:
            skip_conn_channel_spec = skip_conn_channels[encoder_name]
            for block_idx in range(len(skip_conn_channel_spec)):
                x_dim = skip_conn_channel_spec[block_idx]
                if block_idx % 2 == 1:
                    block = CrossAttnBlock(dim=64, heads=4, head_channel=x_dim)
                    self.line_attn_blocks.append(block)
            self.line_attn_blocks = nn.ModuleList(self.line_attn_blocks)

        self.use_attn = use_attn
        self.use_extra_conv = use_extra_conv
        if use_attn or use_extra_conv:
            encoder_channel_spec = encoder_to_last_channels_in_level.get(encoder_name)
            assert (
                encoder_channel_spec is not None
            ), f"Must provide encoder channel spec for {encoder_name=}"

            if do_insert_after:
                for block_idx in range(len(encoder_channel_spec)):
                    x_dim = encoder_channel_spec[block_idx]
                    if use_attn:
                        block = AttentionBasicBlockB(
                            x_dim,
                            x_dim,
                            stride=1,
                            heads=4,
                            window_size=window_size,
                        )
                    elif use_extra_conv:
                        block = nn.Sequential(
                            nn.Conv2d(x_dim, x_dim, kernel_size=1, padding=0),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(x_dim, x_dim, kernel_size=1, padding=0),
                            nn.ReLU(inplace=True),
                        )
                    else:
                        block = nn.Identity()
                    if self.encoder_name == "timm-mobilenetv3_large_100":
                        self.encoder.model.blocks[block_idx] = nn.Sequential(
                            self.encoder.model.blocks[block_idx], block
                        )
                    elif self.encoder_name == "resnet18":
                        layer = getattr(self.encoder, f"layer{block_idx+1}")
                        layer = nn.Sequential(layer, block)
                        setattr(self.encoder, f"layer{block_idx+1}", layer)

            elif self.encoder_name == "timm-mobilenetv3_large_100":
                for block_idx in range(len(self.encoder.model.blocks) - 1):
                    for subblock_idx in range(
                        len(self.encoder.model.blocks[block_idx])
                    ):
                        if not isinstance(
                            self.encoder.model.blocks[block_idx][subblock_idx].se,
                            nn.Identity,
                        ):
                            x_dim = model.encoder.model.blocks[block_idx][subblock_idx].se.conv_reduce.in_channels  # fmt: skip
                        else:
                            x_dim = model.encoder.model.blocks[block_idx][subblock_idx].bn2.bias.numel()  # fmt: skip

                        if use_attn:
                            block = AttentionBasicBlockB(
                                x_dim,
                                x_dim,
                                stride=1,
                                heads=4,
                                window_size=window_size,
                            )
                        elif use_extra_conv:
                            block = nn.Sequential(
                                nn.Conv2d(x_dim, x_dim, kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(x_dim, x_dim, kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                            )
                            if block_idx > 4:
                                block = nn.Sequential(
                                    block,
                                    nn.Conv2d(x_dim, x_dim, kernel_size=1, padding=0),
                                    nn.ReLU(inplace=True),
                                )
                        else:
                            block = nn.Identity()
                        if self.encoder_name == "timm-mobilenetv3_large_100":
                            base_block = model.encoder.model.blocks[block_idx][
                                subblock_idx
                            ]
                            new_se_block = nn.Sequential(
                                block,
                                base_block.se,
                            )
                            base_block.se = new_se_block

        self.decoder = model.decoder
        self.depth_head = model.segmentation_head
        self.df_gap = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.encoder(x)

        if self.attend_line_info:
            if x.shape[1] == 3:
                grayscale = x.mean(dim=1, keepdim=True)
            else:
                grayscale = x
            line_res = self.dlsd(grayscale)
            line_res["df_norm"] = self.df_gap(line_res["df_norm"])
            for i in range(len(self.line_attn_blocks)):
                line_attn_block = self.line_attn_blocks[i]
                features[i * 2 + 1] = line_attn_block(
                    features[i * 2 + 1], line_res["df_norm"]
                )

        decoder_output = self.decoder(*features)

        depth = self.depth_head(decoder_output)

        return depth

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x


if __name__ == "__main__":
    model = DepthModel()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
