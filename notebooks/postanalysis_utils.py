import argparse
import os
import re
from pathlib import Path

import torch
import yaml
from comet_ml.api import API, APIExperiment
from torch.utils.data import DataLoader

from layout_aware_monodepth.data.monodepth import KITTIDataset, NYUv2Dataset
from layout_aware_monodepth.data.transforms import train_transform
from layout_aware_monodepth.train_prototyping import upd_ds_args_with_runtime_args

root_dir = Path(__file__).parent.parent
default_args = argparse.Namespace(
    **yaml.load(open(f"{root_dir}/configs/kitti_ds.yaml"), Loader=yaml.FullLoader)
)

import argparse

import yaml

kitti_args = argparse.Namespace(
    **yaml.load(open(f"{root_dir}/configs/kitti_ds.yaml"), Loader=yaml.FullLoader)
)
nyu_args = argparse.Namespace(
    **yaml.load(open(f"{root_dir}/configs/nyu_ds.yaml"), Loader=yaml.FullLoader)
)
pipe_args = argparse.Namespace(
    **yaml.load(open(f"{root_dir}/configs/pipeline_dev.yaml"), Loader=yaml.FullLoader)
)


def load_ds(args=default_args, mode="train", **extra_kwargs):
    if args.ds_name == "kitti":
        split = extra_kwargs.pop("split", None)
        ds = KITTIDataset(
            args, mode, split=split, transform=train_transform, **extra_kwargs
        )
    else:
        ds = NYUv2Dataset(args, mode, transform=train_transform, **extra_kwargs)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    return ds, dl


def get_latest_ckpt_epoch(exp_name):
    api = API()
    exp_api = api.get(f"kirilllzaitsev/layout-aware-monodepth/{exp_name}")
    ckpt_epochs = [
        int(re.match(r"model_(\d+)\.pth", x["fileName"]).group(1))
        for x in exp_api.get_asset_list(asset_type="all")
        if re.match(r"model_\d+\.pth", x["fileName"])
    ]
    return max(ckpt_epochs)


def load_artifacts_from_comet(exp_name, ckpt_epoch=None):
    artifacts_dir = f"{root_dir}/artifacts/{exp_name}"
    if ckpt_epoch is None:
        ckpt_epoch = get_latest_ckpt_epoch(exp_name)
    weights_path = f"{artifacts_dir}/model_{ckpt_epoch}.pth"
    args_path = f"{artifacts_dir}/train_args.yaml"
    args_not_exist = not os.path.exists(args_path)
    weights_not_exist = not os.path.exists(weights_path)
    if any([args_not_exist, weights_not_exist]):
        api = API(api_key="W5npcWDiWeNPoB2OYkQvwQD0C")
        exp_api = api.get(f"kirilllzaitsev/layout-aware-monodepth/{exp_name}")
        os.makedirs(artifacts_dir, exist_ok=True)
        if args_not_exist:
            asset_id = [
                x
                for x in exp_api.get_asset_list(asset_type="all")
                if "train_args" in x["fileName"]
            ][0]["assetId"]
            api.download_experiment_asset(
                exp_api.id,
                asset_id,
                args_path,
            )
        if weights_not_exist:
            exp_api.download_model(f"depth_model_{ckpt_epoch}", artifacts_dir)
    model_state_dict = torch.load(weights_path)["model_state_dict"]
    train_args = yaml.load(open(args_path), Loader=yaml.FullLoader)
    args = argparse.Namespace(**train_args["args"])
    ds_args = argparse.Namespace(**train_args["ds_args"])
    upd_ds_args_with_runtime_args(args, ds_args)
    return {"model_state_dict": model_state_dict, "args": args, "ds_args": ds_args}


def load_exp_artifacts(exp_name, device, only_args=False):
    artifacts = load_artifacts_from_comet(exp_name, ckpt_epoch=11)
    model_state_dict = artifacts["model_state_dict"]
    args = artifacts["args"]
    ds_args = artifacts["ds_args"]
    ds_args.batch_size = 1
    ds_args.data_path = "/mnt/wext/msc_studies/monodepth_project/data/kitti/kitti-depth"

    if only_args:
        return args, ds_args

    model = load_depth_model(args, device, model_state_dict=model_state_dict)

    return model, args, ds_args


def load_depth_model(args, device, model_state_dict=None, exp_name=None):
    assert not (
        model_state_dict is None and exp_name is None
    ), "Either provide model_state_dict or exp_nameto download the weights from comet"
    if model_state_dict is None:
        artifacts = load_artifacts_from_comet(exp_name, ckpt_epoch=args.ckpt_epoch)
        model_state_dict = artifacts["model_state_dict"]
    from layout_aware_monodepth.model import DepthModel

    img_channels = 1 if args.use_grayscale_img else 3
    model = DepthModel(
        in_channels=img_channels + 1
        if args.line_op in ["concat", "concat_binary"]
        else img_channels,
        use_attn=args.use_attn,
        use_extra_conv=args.use_extra_conv,
        encoder_name=args.backbone,
        do_insert_after=not args.use_attn_before_se,
        decoder_attention_type=args.decoder_attention_type,
        window_size=args.window_size,
        do_attend_line_info=args.do_attend_line_info,
        line_info_feature_map_kwargs=None
        if args.line_embed_channels is None
        else dict(
            image_size=(256, 768),
            patch_size=(8, 16),
            dim=args.line_embed_channels,
            depth=1,
            heads=8,
            channels=args.line_embed_channels,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1,
            dim_head=64,
        ),
        add_df_to_line_info=args.add_df_to_line_info,
        add_df_to_line_info_before_encoder=args.add_df_to_line_info_before_encoder,
        return_deeplsd_embedding=args.return_deeplsd_embedding,
        use_df_to_postproc_depth=getattr(args, "use_df_to_postproc_depth", False),
    )
    model.load_state_dict(model_state_dict, strict=False)
    model.to(device)
    model.dlsd.to(device)
    return model


def load_dnet(
    device,
    pretrained_model="/mnt/wext/msc_studies/monodepth_project/related_work/D-Net/dnet/pretrained_models/efficientnet_b0-kitti_baseline/model_checkpoint",
    params_path=f"{root_dir}/layout_aware_monodepth/models/dnet/arguments_train_kitti_eigen_debug.txt",
):
    from layout_aware_monodepth.models.dnet.dnet import init_dnet

    gpu_id = "0"
    dnet_model = init_dnet(params_path)
    # if os.path.isfile(pretrained_model):
    print("Loading checkpoint '{}'".format(pretrained_model))
    if gpu_id != "-1":
        checkpoint = torch.load(pretrained_model)
    else:
        loc = "cuda:{}".format(gpu_id)
        checkpoint = torch.load(pretrained_model, map_location=loc)

    epoch = checkpoint["epoch"]
    dnet_model.load_state_dict(
        {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
    )
    dnet_model.eval()
    dnet_model.to(device)
    if (
        "history" in checkpoint
    ):  # Allows history to be loaded meaning progress data is preserved.
        history = checkpoint["history"]
    print("Loaded checkpoint '{}' (Epoch {})".format(pretrained_model, epoch))
    return dnet_model
