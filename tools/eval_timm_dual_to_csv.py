#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/eval_timm_dual_to_csv.py
评测：基于 timm 的双视角分类模型（resnet50 / convnext_base / swin_large_* ...）→ 生成 CSV 报告。
- 兼容两种权重形态：
  (A) 共享权重：只有一套 state_dict → 自动复制到 A/B 两路
  (B) 双塔权重：ckpt 含 net_a.* / net_b.* → 自动分别加载
- 兼容训练检查点：优先加载 state_dict_ema（可用 --use_ema 控制），或回退到 state_dict/model
- 键名适配：支持 --strip_prefix（剥离形如 module. / backbone. 的统一前缀）
- 数据：复用你仓库的 datasets.py，只用 val_list；若 DataLoader 返回 (x, gt)，则自动 xb=x 复用

用法示例（见文末）
"""
import argparse, sys, os
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 项目内部
sys.path.append(".")
import datasets as D

# timm 骨干
try:
    import timm
except ImportError as e:
    raise RuntimeError("需要安装 timm：pip install timm") from e


# ========= AP 计算 =========
class AveragePrecisionMeter(object):
    """逐类 AP，再取均值为 mAP（与常见 get_ap 实现一致口径）。"""
    def __init__(self, num_classes: int | None = None):
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.reset()

    def reset(self):
        if self.num_classes is None:
            self._scores = None
            self._targets = None
        else:
            self._scores = [[] for _ in range(self.num_classes)]
            self._targets = [[] for _ in range(self.num_classes)]

    @torch.no_grad()
    def add(self, output, target):
        if not torch.is_tensor(output): output = torch.as_tensor(output)
        if not torch.is_tensor(target): target = torch.as_tensor(target)
        if output.dim() == 1: output = output.view(-1, 1)
        if target.dim() == 1: target = target.view(-1, 1)
        B, C = output.shape
        if self.num_classes is None:
            self.num_classes = C
            self._scores = [[] for _ in range(C)]
            self._targets = [[] for _ in range(C)]
        else:
            if C != self.num_classes:
                raise ValueError("dims mismatch across batches")
        s = output.detach().float().cpu()
        t = (target.detach().float().cpu() >= 0.5).to(torch.int32)
        for k in range(self.num_classes):
            self._scores[k].extend(s[:, k].tolist())
            self._targets[k].extend(t[:, k].tolist())

    def value(self) -> torch.Tensor:
        if self.num_classes is None:
            return torch.tensor([], dtype=torch.float32)
        ap = torch.zeros(self.num_classes, dtype=torch.float32)
        for k in range(self.num_classes):
            scores = torch.tensor(self._scores[k], dtype=torch.float32)
            targets = torch.tensor(self._targets[k], dtype=torch.int32)
            ap[k] = self.average_precision(scores, targets)
        return ap

    @staticmethod
    def average_precision(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if output.numel() == 0: return torch.tensor(0.0, dtype=torch.float32)
        target = (target >= 0.5).to(torch.int32)
        pos_total = int(target.sum().item())
        if pos_total == 0: return torch.tensor(0.0, dtype=torch.float32)
        _, indices = torch.sort(output, descending=True)
        pos = 0.0; tot = 0.0; prec_sum = 0.0
        for idx in indices.tolist():
            tot += 1.0
            if int(target[idx].item()) == 1:
                pos += 1.0
                prec_sum += pos / tot
        return torch.tensor(float(prec_sum / (pos_total + 1e-10)), dtype=torch.float32)


# ========= 双视角 logits 融合 =========
def fuse_logits(la: torch.Tensor, lb: torch.Tensor, how: str = "avg") -> torch.Tensor:
    how = how.lower()
    if how == "add": return la + lb
    if how == "max": return torch.maximum(la, lb)
    return 0.5 * (la + lb)  # avg


# ========= 模型封装 =========
class DualViewHead(nn.Module):
    """两塔同架构（timm），输出 logits；评测时按指定策略融合。"""
    def __init__(self, arch: str, num_classes: int, fuse: str = "avg"):
        super().__init__()
        self.fuse = fuse.lower()
        self.net_a = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        self.net_b = timm.create_model(arch, pretrained=False, num_classes=num_classes)

    def forward(self, xa, xb=None):
        if xb is None: xb = xa
        la = self.net_a(xa)
        lb = self.net_b(xb)
        return fuse_logits(la, lb, self.fuse)


# ========= 权重加载（兼容检查点 + 前缀剥离）=========
def load_state_any(path: Path, prefer_ema: bool = True, strip_prefix: str = "") -> Dict[str, torch.Tensor]:
    """
    兼容 .pth 训练检查点与纯 state_dict：
      - 优先取 state_dict_ema（可由 prefer_ema 控制），否则回退 state_dict，再回退 model
      - 若仍找不到，则尝试把最外层 Tensor 字段“扁平化”为 state_dict
    支持剥离统一前缀（如 'module.' 或 'backbone.'）
    """
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = dict(load_file(str(path)))
    else:
        obj = torch.load(str(path), map_location="cpu")
        sd = None
        if isinstance(obj, dict):
            if prefer_ema and "state_dict_ema" in obj and isinstance(obj["state_dict_ema"], dict):
                sd = obj["state_dict_ema"]
            elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
                sd = obj["state_dict"]
            elif "model" in obj and isinstance(obj["model"], dict):
                sd = obj["model"]
            else:
                sd = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        if sd is None:
            raise RuntimeError(f"无法在 {path} 中找到有效的 state_dict（支持 state_dict_ema/state_dict/model）。")
    if strip_prefix:
        L = len(strip_prefix)
        sd = { (k[L:] if k.startswith(strip_prefix) else k): v for k, v in sd.items() }
    return sd


def try_load_twintower(model: nn.Module, sd: Dict[str, torch.Tensor]) -> int:
    """按 net_a.* / net_b.* 直接加载。返回匹配条数。"""
    msd = model.state_dict()
    to_load = {}
    matched = 0
    for k, v in sd.items():
        if k.startswith("net_a.") or k.startswith("net_b."):
            if k in msd and tuple(msd[k].shape) == tuple(v.shape):
                to_load[k] = v
                matched += 1
    if matched > 0:
        msg = model.load_state_dict(to_load, strict=False)
        print(f"[load][twintower] matched={matched} missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return matched


def shape_safe_load_shared(model: nn.Module, sd: Dict[str, torch.Tensor], add_prefix: str = "") -> Tuple[int,int,int,int]:
    """共享权重：把一套 state_dict 复制到 net_a.*, net_b.*，只加载键名和形状都匹配的条目。"""
    if add_prefix:
        sd = {add_prefix + k: v for k, v in sd.items()}
    msd = model.state_dict()
    to_load = {}
    skipped = 0
    for k, v in sd.items():
        for tower in ("net_a.", "net_b."):
            kk = tower + k
            if kk in msd and tuple(msd[kk].shape) == tuple(v.shape):
                to_load[kk] = v
            else:
                if tower == "net_a.":  # 避免重复计数
                    skipped += 1
    msg = model.load_state_dict(to_load, strict=False)
    matched = len(to_load)
    return matched, len(msg.missing_keys), len(msg.unexpected_keys), skipped


# ========= 类别名 =========
def load_class_names(n: int, path: str) -> List[str]:
    names = [f"class_{i+1:02d}" for i in range(n)]
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = [ln.strip() for ln in f if ln.strip()]
            if len(raw) == n:
                names = raw
    except Exception:
        pass
    return names


# ========= 仅构建 val dataloader（复用你的 datasets.py）=========
def build_val_loader(val_list: str, classes_file: str, input_size: int, batch_size: int, num_classes: int):
    class _A: pass
    a = _A()
    # 你的 build_loaders 需要 train_list 是有效路径，这里传 val_list 以通过检查（训练集不会被使用）
    a.train_list = val_list
    a.val_list = val_list
    a.classes_file = classes_file
    a.input_size = input_size
    a.batch_size = batch_size
    a.num_classes = num_classes
    _, val_loader, class_names = D.build_loaders(a)
    return val_loader, class_names


# ========= 评测主流程 =========
@torch.no_grad()
def evaluate_to_csv(model: nn.Module, val_loader: DataLoader, class_names: List[str],
                    device="cuda", amp=True, csv_path="./output/eval.csv"):
    device = torch.device(device)
    model.eval().to(device)
    apm = AveragePrecisionMeter(num_classes=len(class_names))

    for batch in val_loader:
        # 兼容 (xa, xb, gt) 或 (x, gt)
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                xa, xb, gt = batch
            elif len(batch) == 2:
                xa, gt = batch
                xb = xa
            else:
                raise ValueError(f"Unexpected batch format: len={len(batch)}")
        else:
            raise ValueError("Unexpected batch type")

        xa = xa.to(device, non_blocking=True)
        xb = xb.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda") and amp):
            logits = model(xa, xb)   # 已融合后的 logits
            prob = torch.sigmoid(logits)

        apm.add(prob.detach().cpu(), gt.detach().cpu())

    each_ap = apm.value()                     # [C], 0~1
    mAP = 100.0 * each_ap.mean().item()

    # —— 写 CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = ["mAP(%)"] + [f"{name}_AP(%)" for name in class_names]
    row = [f"{mAP:.3f}"] + [f"{ap*100.0:.3f}" for ap in each_ap.tolist()]
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)
    print(f"[OK] mAP={mAP:.3f}  CSV -> {csv_path}")
    return mAP, each_ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="timm 模型名：resnet50 / convnext_base / swin_large_patch4_window7_224 等")
    ap.add_argument("--ckpt", required=True, help=".pth 或 .safetensors（训练检查点或纯 state_dict 都可）")
    ap.add_argument("--val_list", required=True, help="验证集 txt（双视角或单视角）")
    ap.add_argument("--classes_file", required=True, help="classes.txt")
    ap.add_argument("--num_classes", type=int, default=15)
    ap.add_argument("--input_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fuse", default="avg", choices=["avg", "add", "max"])
    ap.add_argument("--add_prefix", default="", help="共享权重模式下给 ckpt 键名统一加上的前缀（如 'module.' / 'backbone.'）")
    ap.add_argument("--strip_prefix", default="", help="从 ckpt 键名里剥离的前缀（如 'module.' / 'backbone.'）")
    ap.add_argument("--use_ema", default="true", type=str, help="优先加载 state_dict_ema（true/false）")
    ap.add_argument("--csv", required=True, help="结果 CSV 路径")
    args = ap.parse_args()

    # 1) 数据
    val_loader, class_names = build_val_loader(args.val_list, args.classes_file, args.input_size, args.batch_size, args.num_classes)

    # 2) 模型
    model = DualViewHead(args.arch, num_classes=args.num_classes, fuse=args.fuse)

    # 3) 加载权重（先剥前缀，再尝试双塔；若不匹配则共享复制）
    prefer_ema = str(args.use_ema).lower() in ("1","true","t","yes","y")
    sd = load_state_any(Path(args.ckpt), prefer_ema=prefer_ema, strip_prefix=args.strip_prefix)

    matched_twins = try_load_twintower(model, sd)
    if matched_twins == 0:
        matched, missing, unexpected, skipped = shape_safe_load_shared(model, sd, add_prefix=args.add_prefix)
        print(f"[load][shared] matched={matched} missing={missing} unexpected={unexpected} skipped_try={skipped}")

    # 4) 评测 + CSV
    evaluate_to_csv(model, val_loader, class_names, device=args.device, amp=True, csv_path=args.csv)


if __name__ == "__main__":
    main()
