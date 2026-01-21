#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dualview_timm_to_csv.py  (with --train_list)
- 评估双视角 timm 模型；兼容共享权重与双塔权重；将结果输出到 CSV。
"""
import argparse, sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.append(".")
import datasets as D
from engine_finetune import evaluate

try:
    import timm
except ImportError as e:
    raise RuntimeError("需要安装 timm：pip install timm") from e


class DualViewLogitsFuse(nn.Module):
    def __init__(self, arch: str, num_classes: int, fuse: str = "avg"):
        super().__init__()
        self.fuse = fuse.lower()
        self.net_a = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        self.net_b = timm.create_model(arch, pretrained=False, num_classes=num_classes)

    def forward(self, xa, xb=None):
        if xb is None:
            xb = xa
        la = self.net_a(xa)
        lb = self.net_b(xb)
        if self.fuse == "add":
            return la + lb
        elif self.fuse == "max":
            return torch.maximum(la, lb)
        else:
            return 0.5 * (la + lb)


def load_state_any(path: Path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return dict(load_file(str(path)))
    else:
        obj = torch.load(str(path), map_location="cpu")
        if isinstance(obj, dict) and "model" in obj:
            return obj["model"]
        elif isinstance(obj, dict):
            return obj
        raise RuntimeError(f"不支持的 ckpt 对象类型：{type(obj)}")


def try_load_twintower(model: nn.Module, sd: dict):
    msd = model.state_dict()
    to_load = {}
    matched = 0
    for k, v in sd.items():
        if k.startswith("net_a.") or k.startswith("net_b."):
            if k in msd and tuple(v.shape) == tuple(msd[k].shape):
                to_load[k] = v
                matched += 1
    if matched > 0:
        msg = model.load_state_dict(to_load, strict=False)
        print(f"[load][twintower] matched={matched} missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return matched


def shape_safe_load_shared(model: nn.Module, sd: dict, add_prefix: str = ""):
    if add_prefix:
        sd = {add_prefix + k: v for k, v in sd.items()}
    msd = model.state_dict()
    to_load = {}
    for k, v in sd.items():
        for tower in ("net_a.", "net_b."):
            kk = tower + k
            if kk in msd and tuple(v.shape) == tuple(msd[kk].shape):
                to_load[kk] = v
    msg = model.load_state_dict(to_load, strict=False)
    print(f"[load][shared] matched={len(to_load)} missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--csv", required=True, type=str)
    # data
    ap.add_argument("--train_list", default="annotations/DvXray_train.txt")
    ap.add_argument("--val_list", default="annotations/DvXray_val.txt")
    ap.add_argument("--classes_file", default="annotations/classes.txt")
    ap.add_argument("--input_size", default=384, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--num_classes", default=15, type=int)
    # eval
    ap.add_argument("--threshold", default=0.5, type=float)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--fuse", default="avg", type=str, choices=["avg","add","max"])
    # ckpt key 辅助
    ap.add_argument("--add_prefix", default="", type=str)
    args = ap.parse_args()

    device = torch.device(args.device)

    # dataloader（只要 val；但 datasets.build_loaders 需要 train_list 非空路径，所以给一个合法路径即可）
    class _A: pass
    a = _A()
    a.train_list = args.train_list or args.val_list
    a.val_list = args.val_list
    a.classes_file = args.classes_file
    a.input_size = args.input_size
    a.batch_size = args.batch_size
    a.num_classes = args.num_classes
    loaders = D.build_loaders(a)
    _, val_loader, class_names = loaders

    # model
    model = DualViewLogitsFuse(args.arch, num_classes=args.num_classes, fuse=args.fuse).to(device)

    # load ckpt
    sd = load_state_any(Path(args.ckpt))
    matched_twins = try_load_twintower(model, sd)
    if matched_twins == 0:
        shape_safe_load_shared(model, sd, add_prefix=args.add_prefix)

    # eval
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    stats = evaluate(
        data_loader=val_loader,
        model=model,
        device=device,
        criterion=None,
        amp=True,
        threshold=args.threshold,
        class_names=class_names,
        csv_path=args.csv,
        epoch=0,
    )
    print("[eval] results:", stats)
    print(f"[eval] CSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
