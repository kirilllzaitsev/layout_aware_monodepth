import argparse
import glob
import os
import random

import comet_ml
import numpy as np
import torch
import yaml

from layout_aware_monodepth.cfg import cfg


def create_tracking_exp(args) -> comet_ml.Experiment:
    exp_init_args = dict(
        api_key="W5npcWDiWeNPoB2OYkQvwQD0C",
        auto_output_logging="simple",
        auto_metric_logging=True,
        auto_param_logging=True,
        log_env_details=True,
        log_env_host=False,
        log_env_gpu=True,
        log_env_cpu=True,
        log_code=False,
        disabled=args.exp_disabled,
    )
    if args.resume_exp:
        from comet_ml.api import API

        api = API(api_key="W5npcWDiWeNPoB2OYkQvwQD0C")
        exp_api = api.get(f"kirilllzaitsev/layout-aware-monodepth/{args.exp_name}")
        experiment = comet_ml.ExistingExperiment(
            **exp_init_args, experiment_key=exp_api.id
        )
    else:
        experiment = comet_ml.Experiment(
            **exp_init_args, project_name="layout-aware-monodepth"
        )

    for code_file in glob.glob("./*.py"):
        experiment.log_code(code_file)

    return experiment


def log_tags(args, experiment, cfg):
    def add_tag(cond, tag, alt_tag=None):
        if cond:
            tags.append(tag)
        elif alt_tag is not None:
            tags.append(alt_tag)

    tags = [
        args.ds,
        f"{args.line_op}_lines",
        f"filter_{args.line_filter}",
    ]
    tags += args.exp_tags
    add_tag(args.do_overfit, "overfit", "full")
    add_tag(args.use_single_sample, "single_sample")
    add_tag(cfg.is_cluster, "cluster")
    add_tag(args.resume_exp, "resumed")

    experiment.add_tags(tags)


def setup_env(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def save_model(path, epoch, model, optimizer, global_step):
    torch.save(
        {
            "global_step": global_step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return True


def load_ckpt(path, model, optimizer):
    checkpoint = torch.load(path)
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    res = {
        "model": model,
        "optimizer": optimizer,
        "epoch": epoch + 1,
    }
    if "global_step" in checkpoint:
        res["global_step"] = checkpoint["global_step"] + 1
    return res


def load_config(ds_name):
    config_path = f"../configs/{ds_name}_ds.yaml"

    primary_ds_config = yaml.safe_load(open(config_path))
    if cfg.is_cluster:
        aux_ds_config = yaml.safe_load(open(f"../configs/{ds_name}_ds_cluster.yaml"))
        ds_config = {**primary_ds_config, **aux_ds_config}
    else:
        ds_config = primary_ds_config
    ds_args = argparse.Namespace(**ds_config)
    return ds_args
