import skimage
import torch

from layout_aware_monodepth.line_utils import (
    get_deeplsd_pred,
    load_custom_deeplsd,
    load_deeplsd,
)
from layout_aware_monodepth.losses import LineLoss, VPLoss
from layout_aware_monodepth.metrics import calc_metrics, get_metrics
from layout_aware_monodepth.postprocessing import (
    compute_eval_mask,
    postproc_eval_depths,
)
from layout_aware_monodepth.ssl_utils import compute_triplet_loss, get_pose_model


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
        use_vp_loss=False,
        use_line_loss=False,
        use_deeplsd=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        self.device = device
        self.clf_model = clf_model
        self.postproc_block = postproc_block
        self.use_vp_loss = use_vp_loss
        self.use_line_loss = use_line_loss
        if use_vp_loss:
            self.vp_loss = VPLoss()
            self.vp_loss_window_size = args.vp_loss_window_size
            self.vp_loss_scale = args.vp_loss_scale
        if use_line_loss:
            self.line_loss = LineLoss()
            self.line_loss_scale = args.line_loss_scale

        if use_vp_loss or use_line_loss:
            self.dlsd = load_deeplsd().to(device)
        self.do_ssl = getattr(args, "do_ssl", False)
        if self.do_ssl:
            self.pose_model = get_pose_model(self.device)

    def train_step(self, model, batch, criterion, optimizer, epoch, **ssl_kwargs):
        model.train()
        optimizer.zero_grad()
        out = self.model_forward(model, batch)

        res = {}

        if self.do_ssl:
            image0 = batch["image"].to(self.device)
            adj_imgs = batch["adj_imgs"].to(self.device)
            image1 = adj_imgs[:, 0]
            image2 = adj_imgs[:, 1]
            intrinsics = batch["intrinsics"].to(self.device)
            pose01 = self.pose_model.forward(image0, image1)
            pose02 = self.pose_model.forward(image0, image2)
            loss, loss_info = compute_triplet_loss(
                image0=image0,
                image1=image1,
                image2=image2,
                output_depth0=out,
                intrinsics=intrinsics,
                pose01=pose01,
                pose02=pose02,
                w_structure=0.95,
                w_sparse_depth=2.90,
                w_smoothness=0.04,
            )

            res["loss_color"] = loss_info["loss_color"].item()
            res["loss_structure"] = loss_info["loss_structure"].item()
            res["loss_smoothness"] = loss_info["loss_smoothness"].item()
        else:
            y = batch["depth"].to(self.device)
            loss = criterion(out, y)

        res["loss"] = loss.item()
        res["pred"] = out

        if self.use_vp_loss:
            assert epoch is not None, "Epoch must be provided for VP loss"
            if epoch > 0:
                vp_res = self.compute_vp_loss(batch, out, use_depth_as_vp_filter=True)
                loss += self.vp_loss_scale * vp_res["vp_loss"]

                res["vp_loss"] = vp_res["vp_loss"].item()
                res["min_vps_in_batch"] = vp_res["min_vps_in_batch"]
                res["max_vps_in_batch"] = vp_res["max_vps_in_batch"]

        if self.use_line_loss:
            loss_line = self.compute_line_loss(batch, out)
            loss += self.line_loss_scale * loss_line

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        return res

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
                if vp[2] != 0
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
                vp_loss += self.vp_loss(
                    pred[idx], vp, window_size=self.vp_loss_window_size
                )
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

            line_res = get_deeplsd_pred(self.dlsd, x)

            df_embed = line_res["df_embed"]
            concat1 = torch.cat([out, df_embed], dim=1)
            init_shape = concat1.shape[-2:]

            concat = self.resize_tensor(concat1, [i // 2 for i in init_shape])
            postproc_out = self.postproc_block(concat)
            postproc_out = self.resize_tensor(postproc_out, init_shape)

            pred_concat = torch.cat((out, postproc_out), dim=1)
            df = line_res["df_norm"].unsqueeze(1)
            clf_inputs = torch.cat((pred_concat, x, df), dim=1)
            clf_inputs = self.resize_tensor(clf_inputs, [i // 2 for i in init_shape])
            out_confidence = self.clf_model(clf_inputs)
            out_confidence = self.resize_tensor(out_confidence, init_shape)
            out = out * out_confidence[:, 0, :, :].unsqueeze(
                1
            ) + postproc_out * out_confidence[:, 1, :, :].unsqueeze(1)

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
            if self.use_vp_loss:
                vp_res = self.compute_vp_loss(batch, pred, use_depth_as_vp_filter=True)
                result["vp_loss"] = vp_res["vp_loss"].item()
                result["min_vps_in_batch"] = vp_res["min_vps_in_batch"]
                result["max_vps_in_batch"] = vp_res["max_vps_in_batch"]
            if self.use_line_loss:
                loss_line = self.compute_line_loss(batch, pred)
                result["loss_line"] = loss_line.item()
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
