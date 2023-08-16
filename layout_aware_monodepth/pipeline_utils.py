import os
import random

import comet_ml
import numpy as np
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


def log_tags(args, experiment):
    tags = [
        args.ds,
        "overfit" if args.do_overfit else "full",
        f"{args.line_op}_lines",
        f"filter_{args.line_filter}",
    ]
    tags += args.exp_tags
    if args.use_single_sample:
        tags.append("single_sample")

    experiment.add_tags(tags)


def setup_env(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def save_model(path, epoch, model, optimizer):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return True


def load_model(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer
