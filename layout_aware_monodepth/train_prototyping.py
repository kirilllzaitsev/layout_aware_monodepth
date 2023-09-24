import argparse
import json
import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from layout_aware_monodepth.cfg import cfg
from layout_aware_monodepth.data.monodepth import KITTIDataset, NYUv2Dataset
from layout_aware_monodepth.data.transforms import (
    ToTensor,
    test_transform,
    train_transform,
)
from layout_aware_monodepth.extras import EarlyStopper
from layout_aware_monodepth.logging_utils import log_metric, log_params_to_exp
from layout_aware_monodepth.losses import MAELoss, MSELoss, SILogLoss
from layout_aware_monodepth.metrics import RunningAverageDict, calc_metrics
from layout_aware_monodepth.model import DepthModel
from layout_aware_monodepth.pipeline_utils import (
    create_tracking_exp,
    load_ckpt,
    load_config,
    log_tags,
    save_model,
    setup_env,
)
from layout_aware_monodepth.postprocessing import (
    compute_eval_mask,
    postproc_eval_depths,
)
from layout_aware_monodepth.vis_utils import plot_samples_and_preds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self, args, model, optimizer, criterion, train_loader, val_loader, test_loader
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args

    def train_step(self, model, batch, criterion, optimizer):
        model.train()
        y = batch["depth"].permute(0, 3, 1, 2).to(device)
        optimizer.zero_grad()
        out = self.model_forward(model, batch)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        return {"loss": loss.item(), "pred": out}

    def model_forward(self, model, batch):
        x = batch["image"].to(device)
        if self.args.line_op == "concat_embed":
            out = model(x, batch["line_embed"].to(device))
        else:
            out = model(x)
        return out

    def eval_step(self, model, batch, criterion):
        model.eval()
        result = {}
        with torch.no_grad():
            y = batch["depth"].permute(0, 3, 1, 2).to(device)
            pred = self.model_forward(model, batch)
            test_loss = criterion(pred, y)
            result["loss"] = test_loss.item()
            result["pred"] = pred
            pred, y = postproc_eval_depths(
                pred,
                y,
                min_depth=self.args.min_depth_eval,
                max_depth=self.args.max_depth_eval,
            )
            eval_mask = compute_eval_mask(
                y,
                min_depth=self.args.min_depth_eval,
                max_depth=self.args.max_depth_eval,
                crop_type=self.args.crop_type,
                ds_name=self.args.ds,
            )
            metrics = calc_metrics(pred, y, mask=eval_mask)
            return {**result, **metrics}


def run(args):
    setup_env()

    experiment = create_tracking_exp(args)
    exp_dir = (
        f"{cfg.exp_base_dir}/{experiment.name}"
        if cfg.is_cluster
        else f"{cfg.exp_base_dir}/exp"
    )
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment dir: {exp_dir}")

    train_args_path = f"{exp_dir}/train_args.yaml"

    if args.resume_exp:
        previos_args = yaml.safe_load(train_args_path)
        if args.load_previos_args:
            for k, v in previos_args["args"].items():
                setattr(args, k, v)
            for k, v in previos_args["ds_args"].items():
                setattr(ds_args, k, v)
    else:
        with open(train_args_path, "w") as f:
            yaml.dump(
                {"args": vars(args), "ds_args": vars(ds_args)},
                f,
                default_flow_style=False,
            )
    experiment.log_asset(train_args_path)
    os.remove("./train_args_latest.yaml")
    os.symlink(
        train_args_path,
        "./train_args_latest.yaml",
    )
    log_tags(args, experiment, cfg)

    ds_args = load_config(args.ds)
    non_overridden_ds_args = []
    for k, v in vars(args).items():
        if hasattr(ds_args, k):
            if v is None:
                non_overridden_ds_args.append(k)
                continue
            setattr(ds_args, k, v)
        else:
            non_overridden_ds_args.append(k)
    if args.line_embed_channels:
        ds_args.line_embed_channels = args.line_embed_channels
    print(f"Non-overridden ds_args: {non_overridden_ds_args}")

    if args.ds == "kitti":
        ds_cls = KITTIDataset
    else:
        ds_cls = NYUv2Dataset

    train_ds = ds_cls(
        ds_args,
        "train",
        ds_args.split,
        transform=train_transform,
        do_augment=False,
    )

    if args.use_single_sample and args.do_overfit:
        ds_args.batch_size = 1
        args.num_epochs = 100
        args.vis_freq_epochs = 10
        ds_subset = torch.utils.data.Subset(train_ds, range(0, 1))
        train_subset = val_subset = test_subset = ds_subset
        num_workers = 0
    else:
        if args.do_overfit:
            ds_subset = torch.utils.data.Subset(train_ds, range(0, 480))
        else:
            ds_subset = torch.utils.data.Subset(train_ds, range(0, 11_000))
        if args.use_eigen:
            test_subset = ds_cls(
                ds_args,
                "test",
                ds_args.split,
                transform=test_transform,
                do_augment=False,
            )
            train_ds_share = 0.9
            val_ds_share = 0.1
        else:
            train_ds_share = 0.8
            val_ds_share = test_ds_share = 0.1
            test_subset = torch.utils.data.Subset(
                ds_subset,
                range(int(len(ds_subset) * (1 - test_ds_share)), len(ds_subset)),
            )
            test_subset.dataset.transform = test_transform
        num_workers = args.num_workers

        train_ds_len = int(len(ds_subset) * train_ds_share)
        val_ds_len = int(len(ds_subset) * val_ds_share)
        train_subset = torch.utils.data.Subset(ds_subset, range(0, train_ds_len))
        val_subset = torch.utils.data.Subset(
            ds_subset, range(train_ds_len, train_ds_len + val_ds_len)
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=ds_args.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset, batch_size=ds_args.batch_size, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_subset, batch_size=ds_args.batch_size, num_workers=num_workers
    )

    if args.use_single_sample:
        benchmark_batch = next(iter(train_loader))
    else:
        benchmark_paths = json.load(open("../data/data_splits/eval_samples.json"))[
            args.ds
        ]
        benchmark_batch = train_ds.load_benchmark_batch(benchmark_paths)

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
    )
    model.to(device)
    model.dlsd.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # if args.backbone == "timm-mobilenetv3_large_100":
    lr = args.lr or 5e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = SILogLoss()
    early_stopper = EarlyStopper(
        patience=args.num_epochs // 5, min_delta=1e-2, warm_up=3
    )
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=1e-3,
        total_iters=args.num_epochs,
        verbose=True,
    )

    log_params_to_exp(
        experiment,
        cfg.params(),
        "cfg",
    )
    log_params_to_exp(
        experiment,
        vars(ds_args),
        "ds_args",
    )
    log_params_to_exp(
        experiment,
        vars(args),
        "args",
    )
    experiment.log_parameters({"model/num_params": num_params})

    global_step = args.global_step

    if args.resume_exp:
        model, optimizer, args.num_epochs = load_ckpt(
            args.artifacts_path, model, optimizer
        )
    epoch_bar = tqdm(total=args.num_epochs, leave=False)

    trainer = Trainer(
        args, model, optimizer, criterion, train_loader, val_loader, test_loader
    )

    for epoch in range(args.num_epochs):
        train_batch_bar = tqdm(total=len(train_loader), leave=True)
        val_batch_bar = tqdm(total=len(val_loader), leave=True)

        train_metrics_avg = RunningAverageDict()
        val_metrics_avg = RunningAverageDict()
        benchmark_metrics_avg = RunningAverageDict()

        epoch_bar.set_description(f"Epoch {epoch}")

        train_running_losses = []
        val_running_losses = []

        for train_batch in train_loader:
            train_step_res = trainer.train_step(
                model, train_batch, criterion, optimizer
            )
            loss = train_step_res["loss"]
            train_running_losses.append(loss)
            train_batch_bar.update(1)
            train_metrics = {
                f"train_{k}": v for k, v in train_step_res.items() if k not in ["pred"]
            }

            train_batch_bar.set_postfix(**train_metrics)
            global_step += 1
            log_metric(experiment, train_metrics, global_step, prefix="step")
            train_metrics_avg.update(train_metrics)
            # scheduler.step()

        for val_batch in val_loader:
            val_step_res = trainer.eval_step(model, val_batch, criterion)
            val_running_losses.append(val_step_res["loss"])
            val_batch_bar.update(1)
            val_metrics = {
                f"val_{k}": v for k, v in val_step_res.items() if k not in ["pred"]
            }
            val_batch_bar.set_postfix(**val_metrics)
            val_metrics_avg.update(val_metrics)

        epoch_bar.update(1)

        print(f"\nTRAIN metrics:\n{train_metrics_avg}\n")
        print(f"\nVAL metrics:\n{val_metrics_avg}\n")

        log_metric(experiment, train_metrics_avg.get_value(), epoch, prefix="epoch")
        log_metric(experiment, val_metrics_avg.get_value(), epoch, prefix="epoch")

        is_last_epoch = epoch == args.num_epochs - 1
        if (epoch - 1) % args.vis_freq_epochs == 0 or is_last_epoch:
            benchmark_step_res = trainer.eval_step(model, benchmark_batch, criterion)
            benchmark_metrics = {
                f"benchmark_{k}": v
                for k, v in benchmark_step_res.items()
                if k not in ["pred"]
            }
            benchmark_metrics_avg.update(benchmark_metrics)
            log_metric(experiment, benchmark_metrics, epoch, prefix="epoch")
            out = benchmark_step_res["pred"].detach().cpu().permute(0, 2, 3, 1)

            name = "preds/depth"
            for idx in range(len(out)):
                experiment.log_image(
                    out[idx].numpy(),
                    f"{name}_{idx}",
                    step=epoch,
                )

            name = "preds/sample"
            fig = plot_samples_and_preds(
                benchmark_batch,
                out,
                with_depth_diff=True,
                with_colorbar=True,
                max_depth=train_ds.max_depth,
            )
            experiment.log_figure(
                name,
                fig,
                step=epoch,
            )
            plt.close()

            print(f"\nBENCHMARK metrics:\n{benchmark_metrics_avg}\n")

        if (
            args.do_save_model
            and not args.do_overfit
            and ((epoch - 1) % args.save_freq_epochs == 0 or is_last_epoch)
        ):
            save_path = f"{exp_dir}/model_{epoch}.pth"
            save_model(save_path, epoch, model, optimizer)
            experiment.log_model(f"depth_model_{epoch}", save_path, overwrite=False)
            print(f"Saved model to {save_path}")

        train_batch_bar.close()
        val_batch_bar.close()

        test_batch_bar = tqdm(total=len(test_loader), leave=True)
        test_metrics_avg = RunningAverageDict()
        test_running_losses = []
        for test_batch in test_loader:
            test_step_res = trainer.eval_step(model, test_batch, criterion)
            test_running_losses.append(test_step_res["loss"])
            test_batch_bar.update(1)
            test_metrics = {
                f"test_{k}": v for k, v in test_step_res.items() if k not in ["pred"]
            }
            test_metrics_avg.update(test_metrics)
            test_batch_bar.set_postfix(**test_metrics)

        test_batch_bar.close()

        print(f"\nTEST metrics:\n{test_metrics_avg}\n")
        log_metric(experiment, test_metrics_avg.get_value(), epoch, prefix="epoch")

        if early_stopper.early_stop(val_metrics_avg.get_value()["val_loss"]):
            print(
                f"Early stopping. Best val loss: {early_stopper.min_validation_loss}. Current val loss: {val_metrics_avg.get_value()['val_loss']}"
            )
            break

        scheduler.step()

    experiment.add_tags(["finished"])
    experiment.end()


def main():
    import sys

    print("RUNTIME ARGS\n" + " ".join(sys.argv[1:]))
    if cfg.is_cluster:
        print(f"{os.environ['SLURM_JOB_ID']=}")

    parser = argparse.ArgumentParser()

    ds_args_group = parser.add_argument_group("ds_args")
    ds_args_group.add_argument("--min_length", type=int, default=None)
    ds_args_group.add_argument("--use_min_length", action="store_true")
    ds_args_group.add_argument(
        "--ds", type=str, default="nyu", choices=["kitti", "nyu"]
    )
    ds_args_group.add_argument("--do_overfit", action="store_true")
    ds_args_group.add_argument("--use_single_sample", action="store_true")
    ds_args_group.add_argument(
        "--line_op",
        choices=["overlay", "concat", "concat_binary", "concat_embed"],
        default=None,
    )
    ds_args_group.add_argument("--line_filter", default=None)
    ds_args_group.add_argument("--use_grayscale_img", action="store_true")
    ds_args_group.add_argument("--use_eigen", action="store_true")
    ds_args_group.add_argument(
        "--not_load_lines", dest="do_load_lines", action="store_false"
    )
    ds_args_group.add_argument("--crop_type", choices=["garg", "eigen"], default=None)
    ds_args_group.add_argument("--target_shape", nargs=2, type=int, default=None)
    ds_args_group.add_argument(
        "--min_depth_eval",
        type=float,
        default=1e-3,
    )
    ds_args_group.add_argument("--max_depth_eval", type=float, default=10)
    ds_args_group.add_argument("--batch_size", type=int)
    ds_args_group.add_argument("--num_workers", type=int, default=8)

    ops_args_group = parser.add_argument_group("ops_args")
    ops_args_group.add_argument("--save_freq_epochs", type=int, default=2)
    ops_args_group.add_argument("--vis_freq_epochs", type=int, default=1)
    ops_args_group.add_argument("--do_save_model", action="store_true")
    ops_args_group.add_argument("--exp_disabled", action="store_true")
    ops_args_group.add_argument("--exp_tags", nargs="*", default=[])

    ops_args_group.add_argument("--resume_exp", action="store_true")
    ops_args_group.add_argument("--exp_key")
    ops_args_group.add_argument("--artifacts_path")
    ops_args_group.add_argument(
        "--not_load_previos_args", dest="load_previos_args", action="store_false"
    )
    ops_args_group.add_argument("--global_step", type=int, default=0)

    model_args_group = parser.add_argument_group("model_args")
    model_args_group.add_argument("--use_attn", action="store_true")
    model_args_group.add_argument("--use_extra_conv", action="store_true")
    model_args_group.add_argument("--backbone", default="timm-mobilenetv3_large_100")
    model_args_group.add_argument(
        "--decoder_attention_type", default=None, choices=["scse"]
    )
    model_args_group.add_argument("--use_attn_before_se", action="store_true")
    model_args_group.add_argument("--window_size", type=int, default=4)
    model_args_group.add_argument("--do_attend_line_info", action="store_true")
    model_args_group.add_argument("--line_embed_channels", type=int, default=None)
    model_args_group.add_argument("--add_df_to_line_info", action="store_true")
    model_args_group.add_argument(
        "--add_df_to_line_info_before_encoder", action="store_true"
    )
    model_args_group.add_argument(
        "--not_return_deeplsd_embedding",
        dest="return_deeplsd_embedding",
        action="store_false",
    )

    optim_args_group = parser.add_argument_group("optim_args")
    optim_args_group.add_argument("--num_epochs", type=int, default=20)
    optim_args_group.add_argument("--lr", type=float, default=None)

    args = parser.parse_args()
    if args.use_attn_before_se:
        assert (
            args.backbone == "timm-mobilenetv3_large_100"
        ), "No SE blocks in resnet backbone"
    run(args)


if __name__ == "__main__":
    main()
