# engine_finetune.py
# =============== 只用 EMA；兼容多种 batch 结构；评估可写 CSV；autocast 优先 BF16 ===============

import os
import csv
from typing import Tuple, Optional, List

import torch as th
import torch.nn as nn

# 从你的 utils.py 引入日志工具
from utils import MetricLogger, SmoothedValue



# === New: class-level gradient diagnostic ===
import torch as th

def _class_grad_strength(logits: th.Tensor,
                         target: th.Tensor,
                         pos_only: bool = True,
                         alpha: float = 0.75) -> th.Tensor:
    """
    计算每个类别在本 batch 内的“有效梯度强度” G_c（不参与反传，仅统计）。
    多标签 BCE 的一阶梯度对 logit 等于 |sigmoid(z) - y|：
    - 正类：|σ(z)-1| = 1-σ(z)
    - 负类：|σ(z)-0| = σ(z)
    返回: [C] 张量
    """
    with th.no_grad():
        p = th.sigmoid(logits)                         # [B, C]
        pos_mask = (target > 0.5)
        pos_cnt = pos_mask.sum(dim=0).clamp_min(1)
        g_pos = (pos_mask.float() * (1.0 - p)).sum(dim=0) / pos_cnt  # [C]

        if pos_only:
            return g_pos

        neg_mask = (~pos_mask)
        neg_cnt = neg_mask.sum(dim=0).clamp_min(1)
        g_neg = (neg_mask.float() * p).sum(dim=0) / neg_cnt          # [C]
        return alpha * g_pos + (1.0 - alpha) * g_neg


def _grad_var_from(logits: th.Tensor,
                   target: th.Tensor,
                   band: float = 0.0,
                   pos_only: bool = True,
                   alpha: float = 0.75) -> th.Tensor:
    """
    计算类级有效梯度的方差（可加容忍带 band，处于 |diff|<=band 的不计入）。
    返回标量张量（device/logits 同）。
    """
    with th.no_grad():
        g = _class_grad_strength(logits, target, pos_only=pos_only, alpha=alpha)  # [C]
        g_mean = g.mean()
        diff = g - g_mean
        if band > 0.0:
            diff = th.where(diff.abs() > band, diff, th.zeros_like(diff))
        reg = (diff ** 2).mean()
    return reg


# ----------------------------- EMA（可放 CPU） -----------------------------
class SimpleEMA:
    """
    轻量 EMA：维护 '参数名 -> EMA张量' 的字典，可放在 CPU 上节省显存。
    - update(model): 用 model 当前参数进行 EMA 更新
    - copy_to(model): 将 EMA 权重拷到传入模型（常用于验证）
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = "cpu"):
        self.decay = float(decay)
        self.device = th.device(device)
        self.ema_state = {}
        with th.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.ema_state[n] = p.detach().to(self.device, dtype=th.float32).clone()

    @th.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        one_m = 1.0 - d
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n not in self.ema_state:
                self.ema_state[n] = p.detach().to(self.device, dtype=th.float32).clone()
                continue
            tgt = self.ema_state[n]
            src = p.detach().to(tgt.device, dtype=tgt.dtype)
            tgt.mul_(d).add_(src, alpha=one_m)

    @th.no_grad()
    def copy_to(self, model: nn.Module):
        msd = model.state_dict()
        for n, w in self.ema_state.items():
            if n in msd:
                msd[n].copy_(w.to(msd[n].device, dtype=msd[n].dtype))


# ----------------------------- 辅助函数 -----------------------------
def _unpack_samples(samples, device: th.device) -> Tuple[th.Tensor, Optional[th.Tensor], th.Tensor]:
    """
    兼容 batch 结构：
    1) ((xa, xb), y)
    2) (xa, xb, y) / [xa, xb, y]
    3) (xa, y)（单视角）
    4) dict: {'img_a':..., 'img_b':..., 'target':...} 或 {'images':(xa,xb),'target':...}
    """
    xa = xb = target = None

    if isinstance(samples, (tuple, list)):
        if len(samples) == 2:
            x, target = samples
            if isinstance(x, (tuple, list)) and len(x) == 2:
                xa, xb = x
            else:
                xa = x
        elif len(samples) == 3:
            xa, xb, target = samples
        else:
            raise ValueError(f"Unexpected batch tuple length: {len(samples)}")
    elif isinstance(samples, dict):
        if "images" in samples:
            img = samples["images"]
            if isinstance(img, (tuple, list)) and len(img) == 2:
                xa, xb = img
            else:
                xa = img
        else:
            xa = samples.get("img_a", None)
            xb = samples.get("img_b", None)
        target = samples.get("target", None)
    else:
        raise ValueError(f"Unexpected batch structure: type={type(samples)}")

    if xa is None:
        raise ValueError("xa is None after unpacking.")
    if target is None:
        raise ValueError("target is None after unpacking.")

    xa = xa.to(device, non_blocking=True)
    if xb is not None:
        xb = xb.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    return xa, xb, target


def _forward_logits(model: nn.Module, xa: th.Tensor, xb: Optional[th.Tensor]):
    out = model(xa, xb) if xb is not None else model(xa)
    if isinstance(out, (tuple, list)):
        logits = out[0]
    elif isinstance(out, dict) and "logits" in out:
        logits = out["logits"]
    else:
        logits = out
    return logits


def _micro_accuracy(logits: th.Tensor, target: th.Tensor, thresh: float = 0.5):
    with th.no_grad():
        prob = th.sigmoid(logits)
        pred = (prob > thresh).to(target.dtype)
        acc = (pred == target).float().mean()
    return acc


def _f1_scores(logits: th.Tensor, target: th.Tensor, thresh: float = 0.5):
    with th.no_grad():
        prob = th.sigmoid(logits)
        pred = (prob > thresh).to(th.int)
        y = target.to(th.int)

        tp = (pred & y).sum(dim=0).float()
        fp = (pred & (1 - y)).sum(dim=0).float()
        fn = ((1 - pred) & y).sum(dim=0).float()

        denom = (2 * tp + fp + fn).clamp_min(1e-12)
        f1_c = (2 * tp) / denom

        TP = tp.sum(); FP = fp.sum(); FN = fn.sum()
        f1_micro = (2 * TP) / (2 * TP + FP + FN + 1e-12)
        f1_macro = f1_c.mean()
    return f1_micro.item(), f1_macro.item()


def _average_precision_score(scores: th.Tensor, targets: th.Tensor) -> float:
    s = scores.detach().cpu().float()
    y = targets.detach().cpu().float()
    if y.sum() == 0:
        return 0.0
    order = th.argsort(s, descending=True)
    y = y[order]
    tp = y; fp = 1.0 - y
    tp_cum = th.cumsum(tp, dim=0)
    fp_cum = th.cumsum(fp, dim=0)
    recalls = tp_cum / (y.sum() + 1e-12)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-12)
    ap = 0.0; prev_r = 0.0
    for r, p in zip(recalls.tolist(), precisions.tolist()):
        ap += p * max(r - prev_r, 0.0); prev_r = r
    return float(ap)


def _evaluate_multilabel_ap(all_logits: th.Tensor,
                            all_targets: th.Tensor,
                            class_names: Optional[List[str]] = None):
    prob = th.sigmoid(all_logits)
    C = prob.shape[1]
    ap_per_class = []
    for c in range(C):
        ap = _average_precision_score(prob[:, c], all_targets[:, c])
        ap_per_class.append(ap)

    print("---- Per-class AP ----")
    for idx, ap in enumerate(ap_per_class):
        name = class_names[idx] if (class_names and idx < len(class_names)) else f"C{idx}"
        print(f"{name:>18}: AP={ap:.4f}")
    mAP = float(sum(ap_per_class) / max(len(ap_per_class), 1))
    print(f"mAP={mAP:.4f}")
    return ap_per_class, mAP


# ----------------------------- 训练 / 验证 -----------------------------
def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader,
    optimizer: th.optim.Optimizer,
    device: th.device,
    epoch: int,
    drop_path_rate: float = 0.0,  # 只占位，保持签名兼容
    amp: bool = True,
    model_ema: Optional[SimpleEMA] = None,
    print_freq: int = 50,
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("grad_var", SmoothedValue(window_size=20, fmt="{value:.6f}"))
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    scaler = th.cuda.amp.GradScaler(enabled=amp)

    # 优先 BF16；不可用再回落 FP16
    try:
        bf16_ok = th.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    ac_dtype = th.bfloat16 if bf16_ok else th.float16

    header = f"Epoch: [{epoch}]"
    for samples in metric_logger.log_every(data_loader, print_freq, header):
        xa, xb, target = _unpack_samples(samples, device)
        if target.dtype != th.float32:
            target = target.float()

        with th.autocast(device_type='cuda', dtype=ac_dtype, enabled=amp):
            logits = _forward_logits(model, xa, xb)

            # --- 【核心修改】全新的、完全兼容的损失计算逻辑 ---
            # 步骤 1: 永远先计算基础的监督损失 (硬损失)
            # 这确保了 loss_sup 变量在任何情况下都存在！
            loss_sup = criterion(logits, target) if not hasattr(criterion, 'base_criterion') else criterion.base_criterion(logits, target)
            
            # 步骤 2: 将 total_loss 初始化为 loss_sup
            total_loss = loss_sup

            # 步骤 3: 如果是蒸馏模式，再用完整的蒸馏损失覆盖 total_loss
            if 'student_inputs' in criterion.forward.__code__.co_varnames:
                total_loss = criterion(
                    student_outputs=logits,
                    student_inputs=(xa, xb),
                    targets=target
                )
            # ----------------------------------------------------

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if model_ema is not None:
            model_ema.update(model)

        acc1 = _micro_accuracy(logits, target)
        pos_only = getattr(criterion, 'pos_only', True)
        alpha    = getattr(criterion, 'alpha', 0.75)
        band     = getattr(criterion, 'band', 0.0)
        grad_var_t = _grad_var_from(logits, target, band=band, pos_only=pos_only, alpha=alpha)

        # 再更新日志
        metric_logger.update(
            loss=float(total_loss.item()),
            loss_sup=float(loss_sup.item()),
            acc1=float(acc1.item()),
            grad_var=float(grad_var_t.item()),
        )
        for pg in optimizer.param_groups:
            if "lr" in pg:
                metric_logger.update(lr=pg["lr"])
                break

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@th.no_grad()
def evaluate(
    data_loader,
    model: nn.Module,
    device: th.device,
    criterion: Optional[nn.Module] = None,
    amp: bool = True,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
    csv_path: Optional[str] = None,
    epoch: Optional[int] = None,
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")

    try:
        bf16_ok = th.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    ac_dtype = th.bfloat16 if bf16_ok else th.float16

    all_logits = []
    all_targets = []

    for samples in metric_logger.log_every(data_loader, 100, header="Test:"):
        xa, xb, target = _unpack_samples(samples, device)
        if target.dtype != th.float32:
            target = target.float()

        with th.autocast(device_type='cuda', dtype=ac_dtype, enabled=amp):
            logits = _forward_logits(model, xa, xb)
            loss = criterion(logits, target) if (criterion is not None) else th.tensor(0.0, device=device)

        all_logits.append(logits.detach().float().cpu())
        all_targets.append(target.detach().float().cpu())

        metric_logger.update(loss=float(loss.item()))

    logits_cat = th.cat(all_logits, dim=0)
    targets_cat = th.cat(all_targets, dim=0)

    acc1 = _micro_accuracy(logits_cat, targets_cat, thresh=threshold)
    f1_micro, f1_macro = _f1_scores(logits_cat, targets_cat, thresh=threshold)
    per_class_ap, mAP = _evaluate_multilabel_ap(logits_cat, targets_cat, class_names=class_names)

    results = {
        "loss": metric_logger.meters["loss"].global_avg if "loss" in metric_logger.meters else 0.0,
        "acc1": float(acc1.item()),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "mAP": float(mAP),
        "per_class_ap": per_class_ap,
    }

    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        header = ["epoch", "loss", "acc1", "f1_micro", "f1_macro", "mAP"]
        if class_names:
            header += [f"AP_{n}" for n in class_names]
        else:
            header += [f"AP_C{i}" for i in range(logits_cat.shape[1])]

        write_header = (not os.path.isfile(csv_path))
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            row = [
                -1 if epoch is None else int(epoch),
                results["loss"], results["acc1"], results["f1_micro"], results["f1_macro"], results["mAP"],
            ]
            row += [float(x) for x in per_class_ap]
            w.writerow(row)

    return results
