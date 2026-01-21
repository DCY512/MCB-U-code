# models/modules/custom_losses/signals.py
import torch
import torch.nn as nn

def _to_tensor(x, device=None):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach()
    return torch.as_tensor(x, dtype=torch.float32, device=device).detach()

class SignalBus(nn.Module):
    """
    统一“损失侧 -> 注意力侧”的轻量信号总线（EMA 平滑 + detach）。
    训练态：每个 step 由 loss/引擎更新；注意力层读取上一次的 EMA 值（更稳）。
    推理态：固定为最后一次 EMA。
    """
    def __init__(self, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        # [K] 类权（MCB-Convex），[K] 类有效梯度（GE-BCE），标量难度（FALS/DALS）
        self.register_buffer("w_mcb", None)
        self.register_buffer("g_strength", None)
        self.register_buffer("difficulty", torch.tensor(0.0))
        # 双视角分歧（标量），若后续做空间门控，可另行扩展为 map
        self.register_buffer("dv_disagree", torch.tensor(0.0))

    def _ema(self, old, new):
        if new is None:
            return old
        m = self.momentum
        return new.clone() if old is None else old * m + new * (1.0 - m)

    @torch.no_grad()
    def update_mcb_weights(self, w):
        w = _to_tensor(w)
        if w is not None:
            self.w_mcb = self._ema(self.w_mcb, w)

    @torch.no_grad()
    def update_ge_strength(self, g_vec):
        g_vec = _to_tensor(g_vec)
        if g_vec is not None:
            self.g_strength = self._ema(self.g_strength, g_vec)

    @torch.no_grad()
    def update_difficulty(self, d_scalar):
        d_scalar = _to_tensor(d_scalar)
        if d_scalar is not None:
            self.difficulty = self._ema(self.difficulty, d_scalar)

    @torch.no_grad()
    def update_dv_disagree_scalar(self, logits_a, logits_b):
        if logits_a is None or logits_b is None:
            return
        pa = logits_a.detach().sigmoid()
        pb = logits_b.detach().sigmoid()
        s = torch.mean(torch.abs(pa - pb))  # 标量
        self.dv_disagree = self._ema(self.dv_disagree, s)

    def get(self):
        # 读取时不做 clone，避免不必要开销（外部别就地修改）
        return {
            "w_mcb": self.w_mcb,                 # [K] or None
            "g_strength": self.g_strength,       # [K] or None
            "difficulty": self.difficulty,       # scalar
            "dv_disagree": self.dv_disagree,     # scalar
        }

# 单例
signal_bus = SignalBus(momentum=0.9)
