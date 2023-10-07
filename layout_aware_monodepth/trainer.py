import torch

from layout_aware_monodepth.metrics import calc_metrics, get_metrics
from layout_aware_monodepth.postprocessing import (
    compute_eval_mask,
    postproc_eval_depths,
)


class Trainer:
    def __init__(
        self,
        args,
        model,
        optimizer,
        criterion,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        self.device = device

    def train_step(self, model, batch, criterion, optimizer):
        model.train()
        y = batch["depth"].to(self.device)
        optimizer.zero_grad()
        out = self.model_forward(model, batch)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        return {"loss": loss.item(), "pred": out}

    def model_forward(self, model, batch):
        x = batch["image"].to(self.device)
        if self.args.line_op == "concat_embed":
            out = model(x, batch["line_embed"].to(self.device))
        else:
            out = model(x)
        return out

    @torch.no_grad()
    def eval_step(self, model, batch, criterion):
        model.eval()
        result = {}
        with torch.no_grad():
            y = batch["depth"].to(self.device)
            pred = self.model_forward(model, batch)
            test_loss = criterion(pred, y)
            result["loss"] = test_loss.item()
            result["pred"] = pred
            metrics = get_metrics(
                pred,
                y,
                min_depth=self.args.min_depth_eval,
                max_depth=self.args.max_depth_eval,
                crop_type=self.args.crop_type,
                ds_name=self.args.ds,
            )
            return {**result, **metrics}
