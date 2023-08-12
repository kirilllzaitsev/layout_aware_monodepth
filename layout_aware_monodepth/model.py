import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from alternet.models.alternet import AttentionBasicBlockB


class DepthModel(nn.Module):
    # initializers
    def __init__(
        self,
        encoder_name="timm-mobilenetv3_large_100",
        encoder_weights="imagenet",
        decoder_activation="sigmoid",
        decoder_first_channel=256,
        in_channels=3,
        use_attn=False,
        use_extra_conv=False,
    ):
        super().__init__()

        decoder_channels = [decoder_first_channel]
        for i in range(1, 5):
            decoder_channels.append(decoder_first_channel // (2**i))

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=decoder_activation,
            encoder_depth=len(decoder_channels),
            decoder_channels=decoder_channels,
        )

        self.encoder = model.encoder

        self.use_attn = use_attn
        self.use_extra_conv = use_extra_conv

        if use_attn:
            self.attn_blocks = []
            for x_dim in self.encoder.out_channels:
                self.attn_blocks.append(
                    AttentionBasicBlockB(x_dim, x_dim, stride=1, heads=4, window_size=4)
                )

            self.attn_blocks = nn.ModuleList(self.attn_blocks)
        elif use_extra_conv:
            self.attn_blocks = []
            for x_dim in self.encoder.out_channels:
                self.attn_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(x_dim, x_dim, kernel_size=1, padding=0),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(x_dim, x_dim, kernel_size=1, padding=0),
                        nn.ReLU(inplace=True),
                    )
                )

            self.attn_blocks = nn.ModuleList(self.attn_blocks)

        self.decoder = model.decoder
        self.depth_head = model.segmentation_head

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.encoder(x)

        if self.use_attn or self.use_extra_conv:
            for i, feature in enumerate(features):
                features[i] = self.attn_blocks[i](feature)

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
