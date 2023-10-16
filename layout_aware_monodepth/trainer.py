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
    def compute_line_loss(self, batch, out):
        x = batch["image"].to(self.device)
        line_res = get_deeplsd_pred(self.dlsd, x)
        lines = line_res["lines"]
        loss_line = self.line_loss(out, lines)
        return loss_line

    def compute_vp_loss(self, batch, pred, use_depth_as_vp_filter=False):
        x = batch["image"].to(self.device)
        line_res = get_deeplsd_pred(self.dlsd, x)
        vp_loss = torch.tensor(0.0).to(self.device)
        min_vps_in_batch = 100000
        max_vps_in_batch = 0
        for idx in range(len(line_res["vps"])):
            vps = [
                torch.tensor([vp[0] / vp[2], vp[1] / vp[2]]).float()
                for vp in line_res["vps"][idx]
            ]
            h, w = pred.shape[-2:]
            vps = [
                vp
                for vp in vps
                if vp[0] >= 0 and vp[0] < w and vp[1] >= 0 and vp[1] < h
            ]
            if use_depth_as_vp_filter:
                # VPs can be no closer than 15 meters to the camera, computed as a mean depth of a 2x2 window around a VP
                def is_depth_large_enough_for_vp(vp):
                    vp = vp.long()
                    vp_depth_thresh = 15
                    return (
                        torch.mean(
                            pred[idx, :, vp[1] - 2 : vp[1] + 2, vp[0] - 2 : vp[0] + 2]
                        )
                        > vp_depth_thresh
                    )

                vps = [vp for vp in vps if is_depth_large_enough_for_vp(vp)]
            min_vps_in_batch = min(min_vps_in_batch, len(vps))
            max_vps_in_batch = max(max_vps_in_batch, len(vps))
            for vp in vps:
                vp_loss += self.vp_loss(pred[idx], vp)
        return {
            "vp_loss": vp_loss,
            "min_vps_in_batch": min_vps_in_batch,
            "max_vps_in_batch": max_vps_in_batch,
        }

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
