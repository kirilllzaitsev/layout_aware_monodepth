import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.transforms.functional as fn
from alternet.models.alternet import AttentionBasicBlockB, AttentionBlockB

from layout_aware_monodepth.attn_utils import CrossAttnBlock
from layout_aware_monodepth.line_utils import load_custom_deeplsd
from layout_aware_monodepth.vit import ViT

encoder_to_last_channels_in_level = {
    "timm-mobilenetv3_large_100": [16, 24, 40, 80, 112, 160, 960],
    "resnet18": [64, 128, 256, 512],
}
skip_conn_channels = {
    "timm-mobilenetv3_large_100": [3, 16, 24, 40, 112, 960],
    "resnet18": [16, 64, 64, 128, 256, 512],
}


class DepthModel(nn.Module):
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
        line_info_feature_map_kwargs=None,
        add_df_to_line_info=False,
        return_deeplsd_embedding=True,
        add_df_to_line_info_before_encoder=False,
        use_df_as_self_attn_pos_embed=False,
        use_df_as_feature_map=False,
        use_df_to_postproc_depth=False,
        use_deeplsd=False
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
        self.attend_line_info = do_attend_line_info
        self.add_df_to_line_info = add_df_to_line_info
        if (
            do_attend_line_info
            or add_df_to_line_info
            or add_df_to_line_info_before_encoder
            or use_df_to_postproc_depth
            or use_deeplsd
        ):
            self.dlsd = load_custom_deeplsd(
                not self.add_df_to_line_info, return_deeplsd_embedding
            )

        if self.attend_line_info:
            self.line_attn_blocks = self.create_line_attn_blocks(encoder_name)

        self.add_df_to_line_info_before_encoder = add_df_to_line_info_before_encoder
        if self.add_df_to_line_info_before_encoder:
            self.compound_line_info_extractor_window_size = 8
            self.compound_line_info_extractor = AttentionBasicBlockB(
                line_info_feature_map_kwargs["channels"],
                line_info_feature_map_kwargs["channels"],
                stride=1,
                heads=4,
                window_size=self.compound_line_info_extractor_window_size,
                norm=nn.LayerNorm,
                use_pos_emb=True,
            )
            self.proj_x_and_df_to_encoder_input = nn.Conv2d(
                in_channels + line_info_feature_map_kwargs["channels"],
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        use_line_info_as_feature_map = line_info_feature_map_kwargs is not None
        self.use_line_info_as_feature_map = use_line_info_as_feature_map
        if self.use_line_info_as_feature_map:
            self.line_info_extractor = ViT(**line_info_feature_map_kwargs)
            line_info_out_dim = line_info_feature_map_kwargs["dim"]
            self.bottleneck_proj = nn.Conv2d(
                encoder_to_last_channels_in_level[encoder_name][-1],
                encoder_to_last_channels_in_level[encoder_name][-1] - line_info_out_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        self.use_attn = use_attn
        self.use_extra_conv = use_extra_conv
        if use_attn or use_extra_conv:
            encoder_channel_spec = encoder_to_last_channels_in_level.get(encoder_name)
            assert (
                encoder_channel_spec is not None
            ), f"Must provide encoder channel spec for {encoder_name=}"

            if do_insert_after:
                self.embed_attn_after_se_block(
                    window_size, use_attn, use_extra_conv, encoder_channel_spec
                )

            elif self.encoder_name == "timm-mobilenetv3_large_100":
                self.embed_attn_before_se_block(window_size, use_attn, use_extra_conv)
        if self.add_df_to_line_info and not self.add_df_to_line_info_before_encoder:
            x_dim = decoder_channels[-1] + line_info_feature_map_kwargs["channels"]
            block = AttentionBasicBlockB(
                x_dim,
                x_dim,
                stride=1,
                heads=4,
                window_size=window_size,
            )

            self.depth_head = nn.Sequential(
                block, nn.Conv2d(x_dim, 1, kernel_size=3, padding=1)
            )
        else:
            self.depth_head = nn.Sequential(
                nn.Conv2d(decoder_channels[-1], 1, kernel_size=3, padding=1)
            )

        self.use_df_to_postproc_depth = use_df_to_postproc_depth
        if self.use_df_to_postproc_depth:
            self.postproc_depth_layer = self.get_df_self_attn_module()

        self.decoder = model.decoder
        self.df_gap = nn.AdaptiveAvgPool2d((8, 8))
        self.use_df_as_self_attn_pos_embed = use_df_as_self_attn_pos_embed
        self.use_df_as_feature_map = use_df_as_feature_map
        self.final_activation = nn.Sigmoid()

    @staticmethod
    def get_df_self_attn_module(
        in_channels=16 + 1, out_channels=1, postproc_depth_hidden_channels=[64]
    ):
        sa_layers = []

        sa_layers.append(
            AttentionBasicBlockB(
                in_channels,
                postproc_depth_hidden_channels[0],
                stride=1,
                heads=8,
                window_size=8,
                norm=nn.LayerNorm,
                use_pos_emb=True,
            )
        )
        for i in range(1, len(postproc_depth_hidden_channels) - 1):
            sa_layers.append(
                AttentionBasicBlockB(
                    postproc_depth_hidden_channels[i - 1],
                    postproc_depth_hidden_channels[i],
                    stride=1,
                    heads=8,
                    window_size=8,
                    norm=nn.LayerNorm,
                    use_pos_emb=True,
                )
            )
        sa_layers.append(
            AttentionBasicBlockB(
                postproc_depth_hidden_channels[-1],
                out_channels,
                stride=1,
                heads=8,
                window_size=8,
                norm=nn.LayerNorm,
                use_pos_emb=True,
            )
        )
        postproc_depth_layer = nn.Sequential(*sa_layers)
        return postproc_depth_layer

    def forward(self, x, line_info=None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        if self.add_df_to_line_info and self.add_df_to_line_info_before_encoder:
            assert line_info is not None
            line_res = self.get_deeplsd_pred(x)

            df_embed = None
            if self.use_df_as_self_attn_pos_embed:
                df_embed = line_res["df_norm"] / 25
                df_embed = fn.resize(
                    df_embed,
                    (
                        self.compound_line_info_extractor_window_size**2,
                        self.compound_line_info_extractor_window_size**2,
                    ),
                    antialias=True,
                )
            elif self.use_df_as_feature_map:
                df_feature_map = self.convert_df_to_feature_map(line_res["df_norm"])
                line_info += df_feature_map.unsqueeze(1)

            line_info = fn.resize(
                line_info, [i // 2 for i in x.shape[-2:]], antialias=True
            )
            line_info_embed = self.compound_line_info_extractor(
                line_info, pos_embedding=df_embed
            )
            line_info_embed = fn.resize(line_info_embed, x.shape[-2:], antialias=True)
            x = torch.cat([x, line_info_embed], dim=1)
            x = self.proj_x_and_df_to_encoder_input(x)

        features = self.encoder(x)

        if (
            self.use_line_info_as_feature_map
            and not self.add_df_to_line_info_before_encoder
        ):
            if not self.add_df_to_line_info:
                assert line_info is not None
                line_info_embed = self.line_info_extractor(line_info)
                features[-1] = self.bottleneck_proj(features[-1])
                features[-1] = torch.cat(
                    [
                        features[-1],
                        line_info_embed.unsqueeze(-1)
                        .unsqueeze(-1)
                        .repeat((1, 1, features[-1].shape[-2], features[-1].shape[-1])),
                    ],
                    dim=1,
                )

        if self.attend_line_info:
            line_res = self.get_deeplsd_pred(x)
            line_res["df_norm"] = self.df_gap(line_res["df_norm"])
            # TODO: how to make df and features cross-attn sensible? (now it is not)
            for i in range(len(self.line_attn_blocks)):
                line_attn_block = self.line_attn_blocks[i]
                features[i * 2 + 1] = line_attn_block(
                    features[i * 2 + 1], line_res["df_norm"]
                )

        decoder_output = self.decoder(*features)

        if self.add_df_to_line_info and not self.add_df_to_line_info_before_encoder:
            line_res = self.get_deeplsd_pred(x)
            df_embed = self.convert_df_to_feature_map(line_res["df_norm"])
            decoder_output = torch.cat([decoder_output, df_embed], dim=1)

        if self.use_df_to_postproc_depth:
            line_res = self.get_deeplsd_pred(x)
            # df_embed = line_res["df_norm"] / 25
            df_embed = line_res["df_norm"]
            depth = torch.cat([decoder_output, df_embed.unsqueeze(1)], dim=1)
            init_shape = depth.shape[-2:]
            depth = fn.resize(
                depth,
                [i // 2 for i in init_shape],
                antialias=True,
            )
            depth = self.postproc_depth_layer(depth)
            depth = fn.resize(depth, init_shape, antialias=True)
        else:
            depth = self.depth_head(decoder_output)

        depth = self.final_activation(depth)

        depth_scaled = depth * 80

        return depth_scaled

    def create_line_attn_blocks(self, encoder_name):
        line_attn_blocks = []
        skip_conn_channel_spec = skip_conn_channels[encoder_name]
        for block_idx in range(len(skip_conn_channel_spec)):
            x_dim = skip_conn_channel_spec[block_idx]
            if block_idx % 2 == 1:
                block = CrossAttnBlock(dim=64, heads=4, head_channel=x_dim)
                line_attn_blocks.append(block)
        return nn.ModuleList(line_attn_blocks)

    def embed_attn_after_se_block(
        self, window_size, use_attn, use_extra_conv, encoder_channel_spec
    ):
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

    def embed_attn_before_se_block(self, window_size, use_attn, use_extra_conv):
        for block_idx in range(len(self.encoder.model.blocks) - 1):
            for subblock_idx in range(len(self.encoder.model.blocks[block_idx])):
                if not isinstance(
                    self.encoder.model.blocks[block_idx][subblock_idx].se,
                    nn.Identity,
                ):
                    x_dim = self.encoder.model.blocks[block_idx][subblock_idx].se.conv_reduce.in_channels  # fmt: skip
                else:
                    x_dim = self.encoder.model.blocks[block_idx][subblock_idx].bn2.bias.numel()  # fmt: skip

                if use_attn:
                    block = AttentionBasicBlockB(
                        x_dim,
                        x_dim,
                        stride=1,
                        heads=4,
                        window_size=window_size,
                    )
                elif use_extra_conv:
                    # compare how 1x1 conv blocks perform vs attn blocks
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
                    base_block = self.encoder.model.blocks[block_idx][subblock_idx]
                    new_se_block = nn.Sequential(
                        block,
                        base_block.se,
                    )
                    base_block.se = new_se_block

    def get_deeplsd_pred(self, x):
        gray_img = x.mean(dim=1, keepdim=True)
        line_res = self.dlsd(gray_img)
        return line_res

    def convert_df_to_feature_map(self, df):
        embed_channels = 96
        num_bins = embed_channels - 2
        _, edges = torch.histogram(df.flatten().cpu(), bins=num_bins)
        bin_idxs = torch.bucketize(df.flatten().cpu(), edges).reshape(df.shape)
        random_embeds = torch.nn.init.orthogonal_(
            torch.empty(num_bins + 2, embed_channels)
        ).to(df.device)
        new_embeds = random_embeds[bin_idxs].permute(0, 3, 1, 2)
        return new_embeds


if __name__ == "__main__":
    model = DepthModel()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
