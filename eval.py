#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立评估脚本

功能:
- 加载一个已经训练好的模型检查点 (.pth)。
- 在指定的验证/测试数据集上运行评估。
- 输出 mAP, F1-score 等核心性能指标。
- 可以选择加载标准的模型权重或 EMA (指数移动平均) 权重进行评估。
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# --- 从您的项目导入必要的模块 ---
# 确保这些 .py 文件与 eval.py 位于同一项目结构中
from models.convnextv2_dual import ConvNeXtV2Dual
import models.convnextv2 as convnextv2
from engine_finetune import evaluate
import utils as U
import datasets as D

def get_args_parser():
    """定义所有评估所需的命令行参数"""
    parser = argparse.ArgumentParser(description="模型评估脚本")

    # --- 核心参数 ---
    parser.add_argument('--ckpt', required=True, type=str, help="必须提供: 指向要评估的模型检查点文件 (.pth) 的路径")
    parser.add_argument('--val_list', required=True, type=str, help="必须提供: 指向评估数据集的标注文件 (.txt)")
    parser.add_argument('--use_ema', type=U.str2bool, default=True, help="是否使用检查点中保存的 EMA 权重进行评估 (推荐)")

    # --- 模型架构参数 (必须与训练时完全一致!) ---
    parser.add_argument('--model', default='convnextv2_base', help="骨干网络名称")
    parser.add_argument('--fuse_mode', default='xattn', choices=['concat', 'add', 'gated', 'xattn'], help="双视角融合模式")
    parser.add_argument('--fuse_levels', default=['C3','C4','C5'], nargs='+', help="执行融合的特征层级")
    parser.add_argument(
        '--head_type', 
        default='fpn_pan_cbam', 
        choices=['c5','fpn','fpn_fuse','fpn_pan','fpn_pan_se','fpn_pan_cbam','fpn_pan_nonlocal'],
        help="头部结构类型"
    )
    parser.add_argument('--fpn_out_channels', default=256, type=int, help='FPN/PAN 模块的输出通道数')
    parser.add_argument('--xattn_heads', default=4, type=int, help='(仅 xattn 模式) 注意力头数量')
    parser.add_argument('--xattn_reduction', default=4, type=int, help='(仅 xattn 模式) 空间下采样因子')

    # --- 数据与评估参数 ---
    parser.add_argument('--input_size', default=224, type=int, help="模型训练时使用的输入图像尺寸")
    parser.add_argument('--batch_size', default=32, type=int, help="评估时使用的批量大小 (可以比训练时大)")
    parser.add_argument('--classes_file', default='annotations/classes.txt', help="类别名称文件路径")
    parser.add_argument('--num_classes', default=15, type=int, help="类别数量")
    parser.add_argument('--eval_threshold', default=0.5, type=float, help="计算F1分数等指标时使用的概率阈值")
    parser.add_argument('--output_csv', default=None, type=str, help="可选: 将评估结果保存到指定的CSV文件")
    
    # --- 环境参数 ---
    parser.add_argument('--device', default='cuda', help="评估设备 (cuda 或 cpu)")
    parser.add_argument('--num_workers', default=8, type=int, help="数据加载器的工作线程数")

    return parser

def build_model(args):
    """
    根据参数构建模型架构 (从 main_finetune.py 复制而来)
    """
    backbone = convnextv2.convnextv2_base(num_classes=0)

    # 解析 head_type, 分离出基础类型和注意力类型
    head_type_str = getattr(args, "head_type", "c5")
    base_head_type = head_type_str
    attention_type = None
    attention_suffixes = ['_se', '_cbam', '_nonlocal']
    for suffix in attention_suffixes:
        if head_type_str.endswith(suffix):
            base_head_type = head_type_str[:-len(suffix)]
            attention_type = suffix[1:] # remove the '_'
            break

    candidate_kwargs = {
        "backbone": backbone,
        "num_classes": args.num_classes,
        "fuse_mode": args.fuse_mode,
        "fuse_levels": args.fuse_levels,
        "head_type": base_head_type,
        "xattn_heads": args.xattn_heads,
        "xattn_reduction": args.xattn_reduction,
        "fpn_out_channels": args.fpn_out_channels,
        "attention_type": attention_type,
    }
    
    # 过滤掉模型__init__中不存在的参数
    import inspect
    sig = inspect.signature(ConvNeXtV2Dual.__init__)
    valid_keys = set(sig.parameters.keys())
    filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in valid_keys}

    model = ConvNeXtV2Dual(**filtered_kwargs)
    return model

def main(args):
    print("--- 开始评估 ---")
    print(f"评估检查点: {args.ckpt}")
    print(f"评估数据集: {args.val_list}")

    device = torch.device(args.device)

    # 1. 构建与训练时完全相同的模型架构
    print(f"正在构建模型, 骨干: {args.model}, 头部: {args.head_type}, 融合: {args.fuse_mode}...")
    model = build_model(args)
    model.to(device)

    # 2. 加载训练好的权重
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"检查点文件未找到: {args.ckpt}")
    
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    
    state_dict_key = 'model'
    if args.use_ema:
        if 'model_ema' in checkpoint:
            state_dict_key = 'model_ema'
            print("正在加载 EMA 权重...")
        else:
            print("警告: 未找到 EMA 权重, 将使用标准模型权重。")
    else:
        print("正在加载标准模型权重...")

    state_dict = checkpoint[state_dict_key]
    
    # 移除 'module.' 前缀 (如果模型是用 DataParallel 训练的)
    cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 加载权重, strict=True 确保架构完全匹配
    msg = model.load_state_dict(cleaned_state_dict, strict=True)
    print(f"权重加载成功: {msg}")

    # 切换到评估模式
    model.eval()

    # 3. 准备评估数据加载器
    print("正在准备评估数据集...")
    val_dataset = D.DualViewTxtDataset(
        list_file=args.val_list,
        input_size=args.input_size,
        num_classes=args.num_classes,
        train=False  # 确保使用验证集的数据增强
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # 读取类别名称
    class_names = [line.strip() for line in open(args.classes_file)] if os.path.exists(args.classes_file) else None

    # 4. 执行评估
    print(f"\n--- 在 {len(val_dataset)} 张图像上开始评估 ---")
    results = evaluate(
        data_loader=val_loader,
        model=model,
        device=device,
        amp=True, # 使用自动混合精度加速评估
        threshold=args.eval_threshold,
        class_names=class_names,
        csv_path=args.output_csv,
    )

    # 5. 打印最终结果
    print("\n--- 评估完成 ---")
    print(f"  mAP: {results.get('mAP', 0.0):.4f}")
    print(f"  F1-Score (Micro): {results.get('f1_micro', 0.0):.4f}")
    print(f"  F1-Score (Macro): {results.get('f1_macro', 0.0):.4f}")
    print(f"  Accuracy (Micro): {results.get('acc1', 0.0):.4f}")
    if args.output_csv:
        print(f"\n评估结果已保存到: {args.output_csv}")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)