# models/modules/custom_losses/fals_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .signals import signal_bus

# models/modules/custom_losses/fals_loss.py
class FALSLoss(nn.Module):
    def __init__(self, eps: float = 0.1, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.eps = float(eps)
        self.gamma = float(gamma)
        self.reduction = reduction
    def forward(self, logits, targets):
        p  = torch.sigmoid(logits)
        pt = p*targets + (1-p)*(1-targets)
        w  = (1-pt).pow(self.gamma)                # focal_weight
        signal_bus.update_difficulty(w.mean().detach())
        adv = (logits < 0).float()
        y_tilde = (1 - w*self.eps)*targets + w*self.eps*adv
        return F.binary_cross_entropy_with_logits(logits, y_tilde)
