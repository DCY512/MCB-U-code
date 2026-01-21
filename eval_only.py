#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本 (最终版 v1.1)

功能:
- 加载一个训练好的模型检查点 (.pth 文件)。
- 智能地从检查点中恢复训练时的模型配置 (fuse_mode, head_type, attention_type 等)。
- 计算并打印模型的总参数量和可训练参数量。
- 在验证集上运行评估，输出 mAP, per-class AP, F1-micro, F1-macro 等核心指标。
- 可选地将包含所有关键信息的总结性结果追加到一个CSV文件中，便于横向比较。


python eval_only.py ^
  --checkpoint ./output_git/95.16_teacher_v2b384_xattn_fpn_fuse_bs16_git_Xattn_C345_fpn_teacher/checkpoint_best.pth ^
  --val_list ./annotations/DvXray_all.txt ^
  --output_csv ./out_csv/evaluation_teacher_v2b384_xattn_fpn_fuse_git_Xattn_C345.csv
"""
import argparse
import torch
from pathlib import Path
import sys
import csv
import os

# 确保脚本可以找到项目中的其他模块
# (如果 evaluate_model.py 与 main_finetune.py 在同一目录，这通常是可选的)
sys.path.append(str(Path(__file__).parent))

# --- 从您的项目中复用关键模块 ---
# 复用模型构建逻辑
from main_finetune import build_model
# 复用数据加载逻辑
from datasets import build_loaders
# 复用评估循环和指标计算逻辑
from engine_finetune import evaluate
# 复用EMA模型，以评估教师模型的最终状态
from engine_finetune import SimpleEMA

def count_parameters(model: torch.nn.Module):
    """计算并打印模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n--- 模型参数统计 ---")
    print(f"总参数量 (Total): {total_params / 1e6:.2f} M")
    print(f"可训练参数量 (Trainable): {trainable_params / 1e6:.2f} M")
    print("----------------------\n")
    return total_params, trainable_params

def write_to_csv(filepath, data_dict, class_names):
    """将评估结果以追加模式写入指定的CSV文件"""
    # 检查文件是否存在，以决定是否需要写入表头
    file_exists = os.path.isfile(filepath)
    
    # 构建CSV的表头，包含所有需要记录的信息
    header = [
        'checkpoint_path', 'fuse_mode', 'head_type', 'attention_type', 
        'total_params_M', 'trainable_params_M', 
        'mAP', 'f1_micro', 'f1_macro', 'accuracy_micro'
    ]
    # 为每个类别动态添加AP列
    ap_headers = [f'AP_{name}' for name in class_names]
    header.extend(ap_headers)

    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        
        # 如果文件是新建的，则写入表头
        if not file_exists:
            writer.writeheader()
            
        # 准备要写入的数据行字典
        row_data = {
            'checkpoint_path': data_dict['checkpoint'],
            'fuse_mode': data_dict['train_args'].fuse_mode,
            'head_type': f"{data_dict['train_args'].head_type}_{data_dict['train_args'].attention_type}" if getattr(data_dict['train_args'], 'attention_type', None) else data_dict['train_args'].head_type,
            'attention_type': getattr(data_dict['train_args'], 'attention_type', 'N/A'),
            'total_params_M': f"{data_dict['total_params'] / 1e6:.2f}",
            'trainable_params_M': f"{data_dict['trainable_params'] / 1e6:.2f}",
            'mAP': f"{data_dict['eval_stats'].get('mAP', 0.0):.4f}",
            'f1_micro': f"{data_dict['eval_stats'].get('f1_micro', 0.0):.4f}",
            'f1_macro': f"{data_dict['eval_stats'].get('f1_macro', 0.0):.4f}",
            'accuracy_micro': f"{data_dict['eval_stats'].get('acc1', 0.0):.4f}",
        }
        # 添加每个类别的AP值
        if 'per_class_ap' in data_dict['eval_stats'] and len(data_dict['eval_stats']['per_class_ap']) == len(class_names):
            for i, name in enumerate(class_names):
                row_data[f'AP_{name}'] = f"{data_dict['eval_stats']['per_class_ap'][i]:.4f}"
            
        writer.writerow(row_data)
    print(f"\n✅ 评估结果已成功追加到: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="评估一个训练好的双视角模型")
    parser.add_argument('--checkpoint', type=str, required=True, help='指向模型检查点文件的路径 (.pth)')
    parser.add_argument('--val_list', type=str, default=None, help='(可选) 指向验证集文件列表的路径。如果未提供，将使用模型训练时的路径。')
    parser.add_argument('--batch_size', type=int, default=16, help='评估时使用的批量大小。')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载使用的工作线程数。')
    parser.add_argument('--device', type=str, default='cuda', help='评估设备 (cuda 或 cpu)。')
    parser.add_argument('--output_csv', type=str, default=None, help='(可选) 指定一个CSV文件路径，用于保存评估结果。')
    
    cli_args = parser.parse_args()

    # --- 1. 加载检查点和配置 ---
    print(f"📂 正在加载模型检查点: {cli_args.checkpoint}")
    if not Path(cli_args.checkpoint).is_file():
        print(f"❌ 错误: 检查点文件未找到 at {cli_args.checkpoint}")
        return

    checkpoint = torch.load(cli_args.checkpoint, map_location='cpu')
    
    if 'args' not in checkpoint:
        print("❌ 错误: 检查点文件中缺少 'args' 信息，无法自动构建模型。")
        return
        
    # 将字典转换为 Namespace 对象，使其可以像 args 一样通过点号访问
    train_args = argparse.Namespace(**checkpoint['args'])
    
    # 兼容旧的检查点，如果 attention_type 不存在，则设为 None
    if not hasattr(train_args, 'attention_type'):
        train_args.attention_type = None

    print("\n✅ 成功从检查点恢复训练配置:")
    print(f"  - 融合模式 (Fuse Mode): {train_args.fuse_mode}")
    print(f"  - 头部类型 (Head Type): {train_args.head_type}")
    print(f"  - 注意力类型 (Attention): {train_args.attention_type or 'N/A'}")
    
    # 允许命令行参数覆盖部分从检查点中恢复的配置
    train_args.val_list = cli_args.val_list if cli_args.val_list else train_args.val_list
    train_args.batch_size = cli_args.batch_size
    train_args.num_workers = cli_args.num_workers
    
    device = torch.device(cli_args.device)

    # --- 2. 构建模型并加载权重 ---
    print("\n🏗️ 正在根据配置构建模型...")
    model = build_model(train_args).to(device)
    
    # 优先加载 EMA (教师模型) 权重进行评估，这通常是性能最好的版本
    if 'model_ema' in checkpoint:
        print("✨ 检测到 EMA 权重，正在加载 EMA 状态进行评估...")
        ema_model = SimpleEMA(model, device='cpu')
        ema_model.ema_state = checkpoint['model_ema']
        ema_model.copy_to(model)
    else:
        print("正在加载标准模型权重...")
        model.load_state_dict(checkpoint['model'])
    
    model.eval()

    # --- 3. 计算并打印参数量 ---
    total_params, trainable_params = count_parameters(model)

    # --- 4. 准备验证数据集 ---
    print("📦 正在准备验证数据集...")
    try:
        _, data_loader_val, class_names = build_loaders(train_args)
        print(f"  - 验证集样本数: {len(data_loader_val.dataset)}")
        print(f"  - 类别数: {len(class_names)}")
    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")
        return

    # --- 5. 运行评估 ---
    print("\n🚀 开始在验证集上进行评估...")
    
    with torch.no_grad():
        eval_stats = evaluate(
            data_loader=data_loader_val,
            model=model,
            device=device,
            amp=True, # 使用自动混合精度加速评估
            threshold=train_args.eval_threshold,
            class_names=class_names,
            csv_path=None # 评估时不写入每个epoch的csv
        )

    # --- 6. 打印最终的评估报告 ---
    print("\n--- 最终评估报告 ---")
    print(f"模型检查点: {Path(cli_args.checkpoint).name}")
    print("----------------------")
    print(f"📈 mAP (mean Average Precision): {eval_stats.get('mAP', 0.0):.4f}")
    print(f"📈 F1-Score (Micro): {eval_stats.get('f1_micro', 0.0):.4f}")
    print(f"📈 F1-Score (Macro): {eval_stats.get('f1_macro', 0.0):.4f}")
    print(f"📈 准确率 (Micro Accuracy): {eval_stats.get('acc1', 0.0):.4f}")
    print("----------------------")
    # Per-class AP 已经在 evaluate 函数内部打印过了，这里不再重复

    # --- 7. (新增) 如果指定了CSV文件，则写入结果 ---
    if cli_args.output_csv:
        data_to_save = {
            "checkpoint": cli_args.checkpoint,
            "train_args": train_args,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "eval_stats": eval_stats,
        }
        write_to_csv(cli_args.output_csv, data_to_save, class_names)

if __name__ == '__main__':
    main()