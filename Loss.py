import torch
from torch import nn
import numpy
import numpy as np
from utils.process_data import pred_metrics


class PredLoss(nn.Module):
    def __init__(self):
        super(PredLoss, self).__init__()
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out, gt_preds, has_preds):
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)  # (B*N, 6)
        reg = torch.cat([x for x in reg], 0)  # (B*N, 6, 30, 2)
        gt_preds = torch.cat([x for x in gt_preds], 0)  # (B*N, 30, 2)
        has_preds = torch.cat([x for x in has_preds], 0)  # (B*N, 30)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        # loss_out["best_reg_loss"] = zero.clone()
        # loss_out["best_num_reg"] = 0

        num_mods, num_preds = 6, 30
        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0
        # print("mask:",mask)
        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                            (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                            ** 2
                    ).sum(1)
                )
            )

        # dist = []  # 6:[tensor([0.9244, 2.9900, 3.4767, 1.8460, 1.9408, 1.4533, 1.0956, 0.7824, 2.0095,3.7517, 2.4513, 2.8971],...,]
        # for j in range(num_mods):
        #     a = torch.sum(torch.sqrt(torch.sum((reg[:, j] - gt_preds) ** 2, dim=2)) * has_preds, dim=1) / (
        #         has_preds.sum(1))
        #     dist.append(a)

        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)  # (B*N, 6)
        min_dist, min_idcs = dist.min(1)

        """
        min_dist: tensor([1.3409, 1.7954, 0.4266, 3.0010, 0.5336, 2.2302, 1.1922, 0.8161, 0.7629, 0.4237, 1.1059, 4.2381], grad_fn=<MinBackward0>)

        min_idcs: tensor([5, 1, 2, 1, 3, 1, 5, 3, 1, 2, 5, 1])
        """
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)  # torch.arange(12)
        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls

        mask0 = (min_dist < 2.0).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > 0.2
        mgn = mgn[mask0 * mask1]
        mask = mgn < 0.2

        coef = 1.0
        loss_out["cls_loss"] += coef * (
                0.2 * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]  # (B*N,30,2),具有最小位移误差的第K条
        coef = 1.0
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()
        return loss_out


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.pred_loss = PredLoss()

    def forward(self, out, gt_preds, has_preds):
        loss_out = self.pred_loss(out, gt_preds, has_preds)
        loss_out["loss"] = loss_out["cls_loss"] / (
                loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out


class PostProcess(nn.Module):
    def __init__(self):
        super(PostProcess, self).__init__()

    def forward(self, out, gt_preds, has_preds):
        post_out = dict()
        post_out["preds"] = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out["gt_preds"] = [x[0:1].cpu().numpy() for x in gt_preds]
        post_out["has_preds"] = [x[0:1].cpu().numpy() for x in has_preds]
        return post_out

    def append(self, metrics, loss_out, post_out):
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics):
        """Every display-iters print training/val information"""
        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        loss = cls + reg

        preds = np.concatenate(metrics["preds"], 0)  # (B,6,30,2)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)  # (B,30,2)
        has_preds = np.concatenate(metrics["has_preds"], 0)  # (B,30)
        ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        print(
            "loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
            % (loss, cls, reg, ade1, fde1, ade, fde)
        )
