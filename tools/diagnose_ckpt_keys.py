#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, re
import torch
from collections import Counter

# === 配置：你的映射后权重 ===
CKPT = "./weights/convnextv2_base.mapped_to_backbone.safetensors"

# === 构建当前模型（与你训练时保持一致）===
sys.path.append(os.path.abspath("."))  # 保证能 import 本地包
import models.convnextv2 as convnextv2
from models.convnextv2_dual import ConvNeXtV2Dual

model = ConvNeXtV2Dual(backbone=convnextv2.convnextv2_base(num_classes=0),
                       num_classes=15, fuse_mode="add")

msd = model.state_dict()

# === 读取 ckpt ===
from safetensors.torch import load_file
ckpt = dict(load_file(CKPT))

mkeys = set(msd.keys())
ckeys = set(ckpt.keys())

inter = sorted(mkeys & ckeys)
missing = sorted(mkeys - ckeys)       # 模型需要但 ckpt 没有（键名不对或被漏掉）
unexpected = sorted(ckeys - mkeys)    # ckpt 里有但模型没有（通常是命名不符/多余前缀）

print(f"[summary] model={len(mkeys)} keys   ckpt={len(ckeys)} keys")
print(f"[summary] matched={len(inter)}   missing={len(missing)}   unexpected={len(unexpected)}")

# === 统计 blocks 层级的情况 ===
def has_blocks(keys): return any(".blocks." in k for k in keys)
print(f"[probe] model has '.blocks.': {has_blocks(mkeys)}")
print(f"[probe] ckpt  has '.blocks.': {has_blocks(ckeys)}")

# === 看看常见前缀分布（帮助定位集中不匹配位置）===
def bucket(keys):
    buckets = Counter([ ".".join(k.split(".")[:4]) for k in keys ])
    return buckets.most_common(12)

print("\n[top missing buckets (model-only):]")
for b,c in bucket(missing): print(f"  {b}: {c}")

print("\n[top unexpected buckets (ckpt-only):]")
for b,c in bucket(unexpected): print(f"  {b}: {c}")

# === 打印各 20 条样例，辅助人工核对 ===
print("\n[sample missing 20]:")
for k in missing[:20]: print("  ", k)

print("\n[sample unexpected 20]:")
for k in unexpected[:20]: print("  ", k)
