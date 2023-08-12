import argparse
import os
import sys
from pathlib import Path

import comet_ml
import torch


def create_tracking_exp(cfg) -> comet_ml.Experiment:
    experiment = comet_ml.Experiment(
        api_key="W5npcWDiWeNPoB2OYkQvwQD0C",
        project_name="layout-aware-monodepth",
        auto_output_logging="simple",
        auto_metric_logging=True,
        auto_param_logging=True,
        log_env_details=True,
        log_env_host=False,
        log_env_gpu=True,
        log_env_cpu=True,
        log_code=False,
        disabled=cfg.exp_disabled,
    )

    for code_file in [
        "model.py",
        "train.py",
        "train_prototyping.py",
        "metrics.py",
        "losses.py",
        "cfg.py",
        "postprocessing.py",
        "data/monodepth.py",
    ]:
        experiment.log_code(code_file)

    return experiment


def log_tags(args, experiment, cfg):
    tags = [
        args.ds,
        "overfit" if cfg.do_overfit else "full",
        f"{cfg.line_op}_lines",
        f"filter_{args.line_filter}",
    ]
    tags += args.exp_tags
    if args.use_single_sample:
        tags.append("single_sample")

    experiment.add_tags(tags)


def setup_optimizations():
    torch.backends.cudnn.benchmark = True
