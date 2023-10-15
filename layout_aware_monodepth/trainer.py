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
        clf_model=None,
        postproc_block=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        self.device = device
        self.clf_model = clf_model
        self.postproc_block = postproc_block

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
        if self.clf_model is not None:
            assert self.postproc_block is not None

            line_res = model.decoder.get_deeplsd_pred(x)

            df_embed = line_res["df_embed"]
            concat1 = torch.cat([out, df_embed], dim=1)
            init_shape = concat1.shape[-2:]
            import torchvision.transforms.functional as fn

            concat = self.resize_tensor(concat1, [i // 2 for i in init_shape])
            postproc_out = self.postproc_block(concat)
            postproc_out = self.resize_tensor(postproc_out, init_shape)

            pred_concat = torch.cat((out, postproc_out), dim=1)
            df = line_res["df_norm"].unsqueeze(1)
            clf_inputs = torch.cat((pred_concat, x, df), dim=1)
            clf_inputs = self.resize_tensor(clf_inputs, [i // 2 for i in init_shape])
            out_confidence = self.clf_model(clf_inputs)
            out_confidence = self.resize_tensor(out_confidence, init_shape)
            out = (
                out * out_confidence[:, 0, :, :].unsqueeze(1)
                + postproc_out * out_confidence[:, 1, :, :].unsqueeze(1)
            )
        return out

    def resize_tensor(self, tensor, size):
        import torchvision.transforms.functional as fn

        return fn.resize(tensor, size, antialias=True)

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
