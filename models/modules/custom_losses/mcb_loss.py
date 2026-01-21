# 在原文件中新增一个类；保留你现有类不动
import torch
import torch.nn as nn
import torch.nn.functional as F
from .signals import signal_bus


class MCBLoss(nn.Module):
    def __init__(self, tau: float = 1.0, momentum: float = 0.9, reduction: str = "mean"):
        super().__init__()
        self.tau = float(tau)
        self.momentum = float(momentum)
        self.reduction = reduction
        self.register_buffer("ema_w", torch.empty(0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        B, C = logits.shape
        p = torch.sigmoid(logits)
        cls_std = p.std(dim=0, unbiased=False)
        w = torch.softmax(cls_std / max(self.tau, 1e-8), dim=0)

        if self.ema_w.numel() != C:
            self.ema_w = w.detach().clone()
        else:
            self.ema_w.mul_(self.momentum).add_(w.detach(), alpha=1.0 - self.momentum)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss_mat = bce * self.ema_w.unsqueeze(0)
        return loss_mat.mean() if self.reduction == "mean" else loss_mat.sum()


class MCBLossConvex(nn.Module):
    def __init__(self, tau: float = 1.0, w_min: float = 1e-3, momentum: float = 0.9, reduction: str = "mean"):
        super().__init__()
        self.tau = float(tau)
        self.w_min = float(w_min)
        self.momentum = float(momentum)
        self.reduction = reduction
        self.register_buffer("ema_w", torch.empty(0))  # ✅ buffer

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        B, C = logits.shape
        p = torch.sigmoid(logits)                         # [B,C]
        cls_std = p.std(dim=0, unbiased=False)            # [C]
        w = torch.softmax(cls_std / max(self.tau, 1e-8), dim=0)
        w = torch.clamp(w, min=self.w_min)
        w = w / w.sum()      
                                     # ✅ 归一化，防尺度漂移
        signal_bus.update_mcb_weights(w)
        # ✅ EMA（detach 保持当步凸性）
        if self.ema_w.numel() != C:
            self.ema_w = w.detach().clone()
        else:
            self.ema_w.mul_(self.momentum).add_(w.detach(), alpha=1.0 - self.momentum)

        class_weights = self.ema_w.detach()               # ✅ 当步视作常量
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss_mat = bce * class_weights.unsqueeze(0)
        return loss_mat.mean() if self.reduction == "mean" else loss_mat.sum()

