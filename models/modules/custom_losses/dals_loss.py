# models/modules/custom_losses/dals_loss.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .signals import signal_bus

class DALSBCE(nn.Module):
    """
    DALS-BCE：凸版 FALS。
    y_tilde = (1 - w*eps)*y + (w*eps)*(1 - y)，其中 w = ((1 - p_t)^gamma).detach()
    仍是 BCEWithLogits(logits, y_tilde) -> 对 logits 保持凸/连续可微/单调。
    """
    def __init__(self, eps: float = 0.1, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.eps = float(eps)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # p_t: 预测对“真实标签”的置信度
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1.0 - p)              # [B,C]
        # stop-grad 的难度权重（不参与反传，保凸性）
        w = (1.0 - pt).pow(self.gamma).detach()                  # [B,C]
        signal_bus.update_difficulty(w.mean())
        # 朝反类(1-y)做小幅平滑（难样本 w 大，平滑更强）
        y_tilde = (1.0 - w * self.eps) * targets + (w * self.eps) * (1.0 - targets)
        loss = self.bce(logits, y_tilde)                         # [B,C]
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
