#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConvNeXtV2 权重映射脚本 v3.1 (最终修正版)
- 专门适配官方发布的 PyTorch 权重 (.pt/.pth)
- 修正 GRN 参数形状
- 添加 'backbone.' 前缀
- 输出为 safetensors
- [FIX] 确保所有张量在保存前都是 contiguous 的
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict
import torch

try:
    from safetensors.torch import save_file as safetensors_save
except ImportError:
    raise RuntimeError("请先安装 safetensors: pip install safetensors")

def map_key(k: str) -> str:
    """
    对官方发布的 ConvNeXtV2 权重键名进行最小必要转换。
    官方键名与本项目模型结构高度相似，只需处理 DDP 前缀和最终分类头。
    """
    nk = k
    if nk.startswith("module."):
        nk = nk[len("module."):]
    
    if nk.startswith("head.norm."):
        nk = nk.replace("head.norm.", "norm.", 1)
    if nk.startswith("head.fc."):
        nk = nk.replace("head.fc.", "head.", 1)
            
    return nk

def reshape_grn_if_needed(k: str, v: torch.Tensor) -> torch.Tensor:
    """
    官方权重的 GRN gamma/beta 是一维的，需要 reshape 成 (1, 1, 1, C)
    以匹配 timm.layers.GRN 的实现。
    """
    if (".grn.gamma" in k or ".grn.beta" in k) and v.ndim == 1:
        return v.view(1, 1, 1, -1) # .contiguous() will be called later
    return v

def load_state_dict_from_pt(path: Path) -> Dict[str, torch.Tensor]:
    """从 .pt 或 .pth 文件加载 state_dict。"""
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    elif isinstance(obj, dict):
        return obj
    else:
        raise RuntimeError(f"不支持的 ckpt 对象类型：{type(obj)}")

def main():
    parser = argparse.ArgumentParser(description="将官方 ConvNeXtV2 PyTorch 权重映射为项目兼容格式。")
    parser.add_argument("--input", required=True, type=str, help="输入的 .pt 或 .pth 权重文件路径。")
    parser.add_argument("--output", required=True, type=str, help="输出的 .safetensors 文件路径。")
    parser.add_argument("--add_prefix", default="backbone.", type=str, help="为所有键名添加的统一前缀 (例如 'backbone.')。设置为空字符串 '' 则不添加。")
    args = parser.parse_args()

    src_path = Path(args.input)
    dst_path = Path(args.output)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📄 正在读取源权重: {src_path}")

    sd_in = load_state_dict_from_pt(src_path)
    blacklist = ['head.']
    sd_filtered = {}
    for k, v in sd_in.items():
        # 检查当前参数的名称是否以黑名单中的任何前缀开头
        is_blacklisted = any(k.startswith(prefix) for prefix in blacklist)
        if not is_blacklisted:
            sd_filtered[k] = v
    
    print(f"✅ 权重过滤完成: 原始 {len(sd_in)} -> 过滤后 {len(sd_filtered)} (丢弃了 {len(sd_in) - len(sd_filtered)} 个 head 参数)")
    sd_out: Dict[str, torch.Tensor] = {}

    mapped_count = 0
    grn_reshaped_count = 0

    for key, value in sd_filtered.items():
        new_key = map_key(key)
        
        final_value = reshape_grn_if_needed(new_key, value)
        if final_value is not value:
            grn_reshaped_count += 1
        
        if args.add_prefix:
            final_key = args.add_prefix + new_key
        else:
            final_key = new_key
            
        # ==================== FIX ====================
        # 确保张量在保存前是 contiguous 的，以满足 safetensors 的要求
        sd_out[final_key] = final_value.contiguous()
        # =============================================
        
        mapped_count += 1

    print(f"💾 正在保存 {len(sd_out)} 个权重到: {dst_path}")
    safetensors_save(sd_out, str(dst_path))
    
    print("\n✅ 转换完成!")
    print(f"  - 总共处理: {mapped_count} 个权重。")
    print(f"  - GRN 参数形状修正: {grn_reshaped_count} 个。")
    print(f"  - 添加的前缀: '{args.add_prefix}'")

if __name__ == "__main__":
    main()