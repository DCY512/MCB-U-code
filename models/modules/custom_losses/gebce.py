# models/modules/custom_losses/gebce.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .signals import signal_bus

class GEBCELoss(nn.Module):
    def __init__(self, lambda_coef=0.1, pos_only=True, alpha=0.75, ema=True, momentum=0.9, band=0.0,
                 reduction="mean", trainable=False):
        super().__init__()
        self.lambda_coef = float(lambda_coef)
        self.pos_only = bool(pos_only)
        self.alpha = float(alpha)
        self.use_ema = bool(ema)
        self.momentum = float(momentum)
        self.band = float(band)
        self.reduction = reduction
        self.trainable = bool(trainable)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.register_buffer("ema_g", None)
        self.register_buffer("inited", torch.tensor(0, dtype=torch.uint8))

    def _maybe_init_buffers(self, C, device, dtype=torch.float32):
        if (self.ema_g is None) or (self.ema_g.numel() != C):
            self.ema_g = torch.zeros(C, device=device, dtype=dtype)
            self.inited = torch.tensor(1, device=device, dtype=torch.uint8)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        B, C = logits.shape
        per_elem = self.bce(logits, targets)
        loss_bce = per_elem.mean() if self.reduction == "mean" else per_elem.sum()

        # —— 计算类级有效梯度强度 G_c ——（trainable 决定是否允许反传）
        if self.trainable:
            p = torch.sigmoid(logits)                           # 参与反传（非凸）
            pos_mask = (targets > 0.5)
            pos_cnt = pos_mask.sum(dim=0).clamp_min(1)
            g_pos = (pos_mask.float() * (1.0 - p)).sum(dim=0) / pos_cnt
            if self.pos_only:
                g = g_pos
            else:
                neg_mask = (~pos_mask)
                neg_cnt = neg_mask.sum(dim=0).clamp_min(1)
                g_neg = (neg_mask.float() * p).sum(dim=0) / neg_cnt
                g = self.alpha * g_pos + (1.0 - self.alpha) * g_neg
            # EMA 仅作为数值平滑的 buffer，不影响反传路径（buffer 本身无梯度）
            if self.use_ema:
                self._maybe_init_buffers(C, logits.device, dtype=g.dtype)
                self.ema_g.mul_(self.momentum).add_(g.detach(), alpha=(1.0 - self.momentum))
                g_used = g
            else:
                g_used = g
        else:
            with torch.no_grad():                               # 不参与反传（凸性与稳定性更好）
                p = torch.sigmoid(logits)
                pos_mask = (targets > 0.5)
                pos_cnt = pos_mask.sum(dim=0).clamp_min(1)
                g_pos = (pos_mask.float() * (1.0 - p)).sum(dim=0) / pos_cnt
                if self.pos_only:
                    g = g_pos
                else:
                    neg_mask = (~pos_mask)
                    neg_cnt = neg_mask.sum(dim=0).clamp_min(1)
                    g_neg = (neg_mask.float() * p).sum(dim=0) / neg_cnt
                    g = self.alpha * g_pos + (1.0 - self.alpha) * g_neg
                if self.use_ema:
                    self._maybe_init_buffers(C, logits.device, dtype=g.dtype)
                    self.ema_g.mul_(self.momentum).add_(g, alpha=(1.0 - self.momentum))
                    g_used = self.ema_g
                else:
                    g_used = g

        g_mean = g_used.mean()
        signal_bus.update_ge_strength(g_used.detach() if g_used.requires_grad else g_used)
        diff = g_used - g_mean
        if self.band > 0.0:
            diff = torch.where(diff.abs() > self.band, diff, torch.zeros_like(diff))
        reg = (diff ** 2).mean()
        return loss_bce + self.lambda_coef * reg
