#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models.convnextv2 as convnextv2
import models.convnextv1 as convnextv1

from engine_finetune import train_one_epoch, evaluate, SimpleEMA
import utils as U
import datasets as D
import json
import pandas as pd
from models.modules.losses import DistillationLoss
import csv # 确保 csv 已被导入
import inspect
from models.convnextv2_dual import ConvNeXtV2Dual
import torch.multiprocessing as mp



def build_base_criterion(args) -> nn.Module:
    """根据 --base_loss 选择基础监督损失；保持多标签场景默认 BCE。"""
    if args.base_loss == 'bce':
        return nn.BCEWithLogitsLoss()

    elif args.base_loss == 'focal':
        # 你项目里已有实现：models/modules/custom_losses/focal_loss.py
        from models.modules.custom_losses.focal_loss import FocalLoss
        return FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    
    elif args.base_loss == 'mlsm':
        # 输入应为 logits（与 BCEWithLogitsLoss 相同习惯），评估阶段会自行 sigmoid
        return nn.MultiLabelSoftMarginLoss()

    elif args.base_loss == 'fals':
        # 你项目里已有实现：models/modules/custom_losses/fals_loss.py
        from models.modules.custom_losses.fals_loss import FALSLoss
        return FALSLoss(eps=args.fals_eps, gamma=args.fals_gamma, reduction='mean')

    elif args.base_loss == 'mcb':
        # 你项目里已有实现：models/modules/custom_losses/mcb_loss.py
        from models.modules.custom_losses.mcb_loss import MCBLoss
        return MCBLoss(momentum=args.mcb_momentum, reduction='mean')

    elif args.base_loss == 'dals':
        from models.modules.custom_losses.dals_loss import DALSBCE
        return DALSBCE(eps=args.dals_eps, gamma=args.dals_gamma)

    elif args.base_loss == 'mcb_convex':
        from models.modules.custom_losses.mcb_loss import MCBLossConvex
        return MCBLossConvex(tau=args.mcb_tau, w_min=args.mcb_wmin, momentum=args.mcb_momentum)

    elif args.base_loss == 'gebce':
        from models.modules.custom_losses.gebce import GEBCELoss
        return GEBCELoss(
            lambda_coef=args.ge_lambda,
            pos_only=args.ge_pos_only,
            alpha=args.ge_alpha,
            ema=args.ge_ema,
            momentum=args.ge_momentum,
            band=args.ge_band,
            trainable=args.ge_trainable,   # ← 新增
        )


    else:
        # 兜底（向后兼容）
        return nn.BCEWithLogitsLoss()



def append_summary_to_global_log(args, best_metric_value, metric_name, model_total_params, 
                                 class_names, per_class_ap_list):
    """
    将本次实验的最终总结（包含 per-class AP），追加写入到全局日志文件中。
    """
    summary_file_path = Path("experiments_summary_test_2.csv")
    
    # --- 【核心修改】动态构建表头 ---
    headers = [
        'output_dir', 'model', 'aug_mode', 'fuse_mode', 'ahcr_mode', 'attention_config',
        'best_metric_name', 'best_metric_value', 'total_params_M', 'batch_size', 'learning_rate',
    ]
    # 为每个类别都动态添加一个 AP 列
    if class_names and per_class_ap_list:
        ap_headers = [f"AP_{name.replace(' ', '_')}" for name in class_names]
        headers.extend(ap_headers)
    # --------------------------------

    summary_data = {
        'output_dir': args.output_dir,
        'model': args.model,
        'aug_mode': args.aug_mode,
        'fuse_mode': args.fuse_mode,
        'ahcr_mode': args.ahcr_mode if args.fuse_mode == 'ahcr' else 'N/A',
        'attention_config': args.attention_config if args.attention_config else '{}',
        'best_metric_name': metric_name,
        'best_metric_value': f"{best_metric_value:.4f}",
        'total_params_M': f"{model_total_params / 1_000_000:.2f}",
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
    }
    
    # --- 【核心修改】将 per-class AP 数据添加到要写入的行中 ---
    if class_names and per_class_ap_list and len(class_names) == len(per_class_ap_list):
        for i, name in enumerate(class_names):
            summary_data[f"AP_{name.replace(' ', '_')}"] = f"{per_class_ap_list[i]:.4f}"
    # ----------------------------------------------------

    try:
        file_exists = summary_file_path.is_file()
        with open(summary_file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary_data)
        print(f"📈 最终结果（含Per-Class AP）已成功追加到总成绩表: {summary_file_path}")
    except Exception as e:
        print(f"❌ 写入总成绩表时发生错误: {e}")


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def _load_finetune_weights(model: nn.Module, ckpt_path: str, prefix: str = '') -> None:
    if not ckpt_path:
        return
    p = Path(ckpt_path)
    if not p.exists():
        print(f"[finetune] file not found: {ckpt_path}")
        return

    sd = None
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(p))
    else:
        obj = torch.load(str(p), map_location="cpu")
        sd = obj["model"] if (isinstance(obj, dict) and "model" in obj) else obj

    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if prefix and nk.startswith(prefix):
            nk = nk[len(prefix):]
        cleaned[nk] = v

    try:
        mount = getattr(args, "model_mount", "")
    except NameError:
        mount = ""
    if mount:
        cleaned = {(mount + k): v for k, v in cleaned.items()}

    msd = model.state_dict()
    to_load = {}
    skipped_shape = []
    for k, v in cleaned.items():
        if k in msd and tuple(v.shape) == tuple(msd[k].shape):
            to_load[k] = v
        elif k in msd:
            skipped_shape.append((k, tuple(v.shape), tuple(msd[k].shape)))

    msg = model.load_state_dict(to_load, strict=False)
    print(f"[finetune] loaded={len(to_load)}  skipped_shape={len(skipped_shape)}  "
          f"missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")
    
    # === 打印缺失/意外键（最多前 200 条，避免刷屏） ===
    _missing = list(getattr(msg, 'missing_keys', []))
    _unexpected = list(getattr(msg, 'unexpected_keys', []))

    if _missing:
        print("[finetune] missing keys (first 200):")
        for k in _missing[:200]:
            print("  -", k)

    if _unexpected:
        print("[finetune] unexpected keys (first 200):")
        for k in _unexpected[:200]:
            print("  -", k)

    # === 方便排查：把完整清单保存到输出目录（若能获取） ===
    # 尝试从环境变量或常见变量里拿 output_dir；拿不到就落到当前目录
    out_dir = os.environ.get("OUTPUT_DIR_HINT", "")
    try:
        # 如果主程序在调用前设置过 args.output_dir，这里也许能取到
        out_dir = out_dir or getattr(globals().get('args', None), 'output_dir', '')
    except Exception:
        pass

    save_root = out_dir if out_dir else "."
    try:
        os.makedirs(save_root, exist_ok=True)
        with open(os.path.join(save_root, "finetune_missing_keys.txt"), "w") as f:
            for k in _missing:
                f.write(k + "\n")
        with open(os.path.join(save_root, "finetune_unexpected_keys.txt"), "w") as f:
            for k in _unexpected:
                f.write(k + "\n")
        print(f"[finetune] 已将缺失/意外键清单写入到: {save_root}/finetune_*_keys.txt")
    except Exception as e:
        print(f"[finetune] ⚠️ 保存缺失/意外键清单失败: {e}")

    if skipped_shape:
        print("[finetune] first few shape-mismatch keys:")
        for i, (k, s_ckpt, s_model) in enumerate(skipped_shape[:10]):
            print(f"  - {k}: ckpt{s_ckpt} vs model{s_model}")


def _try_build_loaders_with_project(args):
    builder_names = ["build_loaders", "build_dataloaders", "create_loaders", "create_dataloaders"]
    for name in builder_names:
        if hasattr(D, name):
            return getattr(D, name)(args)
    if hasattr(D, "XrayMultiLabelList"):
        train_ds = D.XrayMultiLabelList(args.train_list, args.classes_file, is_train=True,
                                        dual_view=args.dual_view, input_size=args.input_size)
        val_ds = D.XrayMultiLabelList(args.val_list, args.classes_file, is_train=False,
                                      dual_view=args.dual_view, input_size=args.input_size)
        
        # 优化后的 DataLoader 配置，充分利用共享内存
        train_loader = DataLoader(
            train_ds, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,  # 关键：启用固定内存
            prefetch_factor=2,  # 预取批次
            persistent_workers=True,  # 保持worker进程
            multiprocessing_context='spawn',  # 使用spawn方式
            drop_last=True
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            multiprocessing_context='spawn'
        )
        return train_loader, val_loader, getattr(train_ds, "class_names", None)
    raise RuntimeError("datasets.py 缺少构建函数（build_loaders/...），请保留工程里的数据集逻辑。")


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=True)
    # basic
    parser.add_argument('--model', default='convnextv2_base')
    parser.add_argument('--input_size', default=384, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--drop_path', default=0.2, type=float)
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', type=int, default=8)

    # init weights
    parser.add_argument('--finetune', default='')
    parser.add_argument('--model_prefix', default='', help="加前缀")
    parser.add_argument("--model_mount", type=str, default="", help="可选：统一挂在到某子模块前")

    # data
    parser.add_argument('--dual_view', default=True, type=U.str2bool)
    parser.add_argument('--train_list', default='annotations/DvXray_train.txt')
    parser.add_argument('--val_list',   default='annotations/DvXray_val.txt')
    parser.add_argument('--classes_file', default='annotations/classes.txt')
    parser.add_argument('--num_classes', default=15, type=int)
    parser.add_argument('--multi_label', default=True, type=U.str2bool)
    parser.add_argument('--eval_threshold', default=0.5, type=float)

    # backbone feats
    parser.add_argument('--return_intermediate', default=False, type=U.str2bool)
    parser.add_argument('--out_indices', default=[1,2,3], nargs='+', type=int)

    # dual-fuse head（新增 xattn 选项 + 参数）
    parser.add_argument('--fuse_mode', default='concat', choices=['concat', 'add', 'gated', 'xattn', 'ahcr'])
    parser.add_argument('--fuse_levels', default=['C3','C4','C5'], nargs='+')
    # dual-fuse head（新增 fpn_fuse 选项）
    parser.add_argument('--head_type', default='c5', type=str, help="Type of head for feature aggregation: c5, fpn, fpn_fuse, fpn_pan")
    parser.add_argument('--attention_config', type=str, default=None,
                    help='JSON string for complex attention configurations. '
                         'Example: \'{"N3": "eca", "N4": ["se", "cbam"]}\'')

    parser.add_argument('--xattn_heads', default=4, type=int, help='仅 fuse_mode=xattn 时使用')
    parser.add_argument('--xattn_reduction', default=4, type=int, help='空间下采样因子，2/4/8')
    parser.add_argument('--fpn_out_channels', default=256, type=int, help='FPN输出通道数')  # 新增参数
    
    # ... 其他参数 ...
    parser.add_argument('--patience', type=int, default=0,
                    help='Enable early stopping if validation metric does not improve for this many epochs. '
                         'Default 0 to disable.')
    parser.add_argument('--ahcr_mode', default='intra_level', choices=['intra_level', 'inter_level'],
                        help="Defines the hierarchical strategy for AHCR fusion. Only used if fuse_mode is 'ahcr'.")
    # teacher/EMA
    parser.add_argument('--teacher_mode', default=True, type=U.str2bool)
    parser.add_argument('--ema_decay', default=0.9999, type=float)
    parser.add_argument('--ema_device', default='cpu')
    parser.add_argument('--fsdp_cpu_offload', default=False, type=U.str2bool)
    # 续训参数
    parser.add_argument('--resume', default='', help='从检查点恢复训练 (checkpoint_last.pth 或 checkpoint_best.pth)')
    parser.add_argument('--resume_epoch', default=-1, type=int, help='从指定epoch开始（默认自动检测）')
    parser.add_argument('--resume_optimizer', default=True, type=U.str2bool, help='是否恢复优化器状态')
    parser.add_argument('--resume_scheduler', default=True, type=U.str2bool, help='是否恢复学习率调度器')


    # 蒸馏
    parser.add_argument('--use_distillation', type=U.str2bool, default=False)
    parser.add_argument('--teacher_model', type=str, default='convnext_small')
    parser.add_argument('--teacher_weights', type=str, default='')
    parser.add_argument('--kd_mode', type=str, default='logits', choices=['logits', 'dkd'])
    parser.add_argument('--distillation_alpha', type=float, default=0.5, help="Hard loss weight.")
    parser.add_argument('--distillation_tau', type=float, default=2.0)
    parser.add_argument('--distill_feature_layers', type=str, nargs='+', default=None)
    parser.add_argument('--distillation_beta', type=float, default=0.0, help="Feature loss weight.")
    parser.add_argument('--dkd_alpha', type=float, default=1.0)
    parser.add_argument('--dkd_beta', type=float, default=8.0)

    # --- Base Loss 选择（默认 bce，向后兼容） ---
    parser.add_argument('--base_loss', type=str, default='bce',
                    choices=['bce','mlsm','focal','fals','mcb','gebce','dals','mcb_convex'],
                    help='选择基础监督损失：bce / mlsm / focal / fals / mcb / gebce / dals / mcb_convex')

    # Focal Loss 超参
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=float, default=None)  # 可为 None 或 float

    # FALS（焦点对抗性标签平滑）超参
    parser.add_argument('--fals_eps', type=float, default=0.1)
    parser.add_argument('--fals_gamma', type=float, default=2.0)

    # MCB（元加权类别平衡）超参
    parser.add_argument('--mcb_momentum', type=float, default=0.9)

    # --- GE-BCE 超参（默认稳健值） ---
    parser.add_argument('--ge_lambda', type=float, default=0.1,
                        help='GE-BCE: class-level gradient equalization strength')
    parser.add_argument('--ge_pos_only', type=U.str2bool, default=True,
                        help='GE-BCE: use positives only to compute G_c')
    parser.add_argument('--ge_alpha', type=float, default=0.75,
                        help='GE-BCE: weight for positives when pos_only is False')
    parser.add_argument('--ge_ema', type=U.str2bool, default=True,
                        help='GE-BCE: EMA smoothing over G_c')
    parser.add_argument('--ge_momentum', type=float, default=0.9,
                        help='GE-BCE: EMA momentum')
    parser.add_argument('--ge_band', type=float, default=0.0,
                        help='GE-BCE: tolerance band; diffs within band are not penalized')

    # DALS（凸版 FALS）
    parser.add_argument('--dals_eps',   type=float, default=0.1)
    parser.add_argument('--dals_gamma', type=float, default=2.0)

    # MCB（凸化）
    parser.add_argument('--mcb_tau',  type=float, default=1.0)
    parser.add_argument('--mcb_wmin', type=float, default=1e-3)

    # GE-BCE：是否让正则参与反传（true=非凸；false=仅诊断）
    parser.add_argument('--ge_trainable', type=U.str2bool, default=False,
                        help='GE 正则是否参与反传（true=非凸；false=仅诊断）')
    # 数据增强
    parser.add_argument('--aug_mode', default='standard',
                    choices=['none','standard',
                             'conditional','conditional_1','conditional_2','conditional_3','conditional_4',
                             'rand_aug','trivial_aug'],
                    help="Select data augmentation strategy.")
    parser.add_argument('--rand_aug_n', type=int, default=2, help="Hyperparameter N for RandAugment.") 
    parser.add_argument('--rand_aug_m', type=int, default=9, help="Hyperparameter M for RandAugment (0-30).")
    # CSV eval
    parser.add_argument('--eval_csv', default='')
    # 新增：评估时将 per-class AP 列按 AP 值排序写入 CSV（默认 False）
    parser.add_argument('--eval_csv_sort_classes', default=False, type=U.str2bool,
                        help="If true, sort per-class AP columns by descending AP when writing CSV.")
    # 新增：每个样本输出 top-k 预测（0 表示不输出）
    parser.add_argument('--eval_csv_topk', default=0, type=int,
                        help="If >0, write per-sample top-k predictions (class:score) to a separate CSV per epoch.")
    return parser


def build_model(args):
    

    # --- 【核心修改】智能模型构建逻辑 ---
    backbone_builder = None
    # 1. 优先在 convnextv2 模块中查找
    if hasattr(convnextv2, args.model):
        backbone_builder = getattr(convnextv2, args.model)
        print(f"✅ 从 [ConvNeXtV2] 模块中成功找到模型构建器: {args.model}")
    # 2. 如果 V2 中没有，再去 convnextv1 模块中查找
    elif hasattr(convnextv1, args.model):
        backbone_builder = getattr(convnextv1, args.model)
        print(f"✅ 从 [ConvNeXtV1] 模块中成功找到模型构建器: {args.model}")
    
    if backbone_builder is None:
        raise ValueError(f"错误: 在 models/convnextv2.py 或 models/convnextv1.py 中均未找到名为 '{args.model}' 的模型函数。")
    # -------------------------------------------

    # 尝试用最少的参数初始化backbone，以避免冲突
    # 这对于加载 V1 的 timm 风格模型很重要
    try:
        backbone = backbone_builder(num_classes=0)
    except TypeError:
        backbone = backbone_builder()

    sig = inspect.signature(ConvNeXtV2Dual.__init__)
    valid_keys = set(sig.parameters.keys())

    out_idx = tuple(getattr(args, "out_indices", (1, 2, 3)))
    fuse_kw = getattr(args, "fuse_mode", "add")
    
    # --- 【关键修正】清理了旧的、无用的注意力解析逻辑 ---
    # 现在直接使用 head_type
    base_head_type = getattr(args, "head_type", "c5")
    
    # 解析 JSON 格式的注意力配置
    parsed_attention_config = None
    if getattr(args, "attention_config", None):
        try:
            parsed_attention_config = json.loads(args.attention_config)
            print("✅ 成功解析注意力配置:", parsed_attention_config)
        except json.JSONDecodeError:
            raise ValueError(f"错误: 解析 --attention_config 的 JSON 字符串失败: {args.attention_config}")
    
    # 构造传递给 ConvNeXtV2Dual 的参数字典
    candidate_kwargs = {
        "backbone": backbone,

        "num_classes": args.num_classes,
        "fuse_mode": fuse_kw,
        "return_intermediate": getattr(args, "return_intermediate", False),
        "out_indices": out_idx,
        "fuse_levels": getattr(args, "fuse_levels", None),
        "head_type": base_head_type,
        "xattn_heads": getattr(args, "xattn_heads", 4),
        "xattn_reduction": getattr(args, "xattn_reduction", 4),
        "fpn_out_channels": getattr(args, "fpn_out_channels", 256),
        "attention_config": parsed_attention_config,
        "ahcr_mode": getattr(args, "ahcr_mode", "intra_level"),
    }
    
    # 过滤掉不在 ConvNeXtV2Dual.__init__ 参数列表中的键
    filtered = {k: v for k, v in candidate_kwargs.items() if k in valid_keys}

    model = ConvNeXtV2Dual(**filtered)
    return model


def _safe_evaluate(data_loader_val, model_to_eval, device, amp=True, class_names=None, threshold=None, csv_path=None):
    try:
        return evaluate(data_loader_val, model_to_eval, device, amp=amp,
                        class_names=class_names, threshold=threshold, csv_path=csv_path)
    except TypeError:
        try:
            return evaluate(data_loader_val, model_to_eval, device, amp=amp,
                            class_names=class_names, threshold=threshold)
        except TypeError:
            try:
                return evaluate(data_loader_val, model_to_eval, device, amp=amp)
            except TypeError:
                return evaluate(data_loader_val, model_to_eval, device)

def _load_resume_checkpoint(model, optimizer, model_ema, checkpoint_path, args):
    """加载续训检查点"""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return 0, -1.0, 0
    
    print(f"📂 加载续训检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model'])
    
    # 加载优化器状态
    if args.resume_optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("✅ 优化器状态已恢复")
    
    # 加载EMA模型状态
    if model_ema is not None and 'model_ema' in checkpoint:
        # 需要重新创建EMA状态
        model_ema.ema_state = checkpoint['model_ema']
        print("✅ EMA模型状态已恢复")
    
    # 获取起始epoch和最佳指标
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_metric = checkpoint.get('metric', {}).get('mAP', -1.0)
    # --- [早停] 从检查点加载早停计数器 ---
    epochs_since_best = checkpoint.get('epochs_since_best', 0)
    
    print(f"✅ 从 {checkpoint_path} 成功续训。")
    print(f"   - 起始轮次: {start_epoch}")
    print(f"   - 已达最佳 mAP: {best_metric:.4f}")
    print(f"   - 早停计数器状态: {epochs_since_best}")

    # --- [早停] 返回包含计数器的新元组 ---
    return start_epoch, best_metric, epochs_since_best





def main(args):
    print(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # 设置共享内存策略
    
    try:
        mp.set_sharing_strategy('file_system')
        print("✅ 共享内存策略设置为 'file_system'")
    except Exception as e:
        print(f"⚠️ 设置共享内存策略时出错: {e}")
    
    # 设置 PyTorch 内存优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.max_split_size_mb = 512
    
    # 启用 CUDA 内存优化
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)  # 使用 90% GPU 内存
        print("✅ CUDA 内存优化已启用")

    loaders = _try_build_loaders_with_project(args)
    if len(loaders) == 3:
        data_loader_train, data_loader_val, class_names = loaders
    else:
        data_loader_train, data_loader_val = loaders
        class_names = None

# --- 【核心修改】重构的、逻辑正确的模型与损失创建流程 ---

    # 1. 永远先创建学生模型
    print(" assembling student model...")


    # 先构建学生模型...（保持不变）
    model = build_model(args).to(device)
    if args.finetune:
        print(f"Load pre-trained student weights from: {args.finetune}")
        _load_finetune_weights(model, args.finetune, prefix=args.model_prefix or '')

    # === 新增：无论是否蒸馏，先构建“基础监督损失” ===
    base_criterion = build_base_criterion(args)
    print(f"[BaseLoss] Using {args.base_loss}  ->  {base_criterion.__class__.__name__}")

    if args.use_distillation:
        print("🔥 Knowledge distillation mode enabled!")
        # 组装教师模型（保持你原来的逻辑）
        teacher_build_args = argparse.Namespace(**vars(args))
        teacher_build_args.model = args.teacher_model
        teacher_model = build_model(teacher_build_args).to(device)

        if args.teacher_weights:
            print(f"   - Loading teacher weights from: {args.teacher_weights}")
            _load_finetune_weights(teacher_model, args.teacher_weights)
        else:
            print("   - ⚠️ WARNING: No teacher weights provided. Teacher will use random weights.")

        # 用“基础监督损失”作为蒸馏包装里的 base_criterion
        from models.modules.losses import DistillationLoss
        criterion = DistillationLoss(
            base_criterion=base_criterion,     # ★ 关键：这里换成可选的基础损失
            student_model=model,
            teacher_model=teacher_model,
            kd_mode=args.kd_mode,
            alpha=args.distillation_alpha,
            beta=args.distillation_beta,
            dkd_alpha=args.dkd_alpha,
            dkd_beta=args.dkd_beta,
            tau=args.distillation_tau,
            feature_layers=args.distill_feature_layers,
            adapter_configs=None,
        )
        print(f"   - DistillationLoss ready (kd_mode={args.kd_mode}, alpha={args.distillation_alpha}, tau={args.distillation_tau})")
    else:
        print("🔷 Standard training mode.")
        # 非蒸馏直接用基础损失
        criterion = base_criterion

    print("Criterion =", criterion.__class__.__name__)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_ema = None
    if args.teacher_mode:
        model_ema = SimpleEMA(model, decay=args.ema_decay, device=args.ema_device)
        print(f"[teacher_mode] EMA enabled with decay={args.ema_decay}, device={args.ema_device}")

    # ========== 续训逻辑 ==========
    start_epoch = 0
    best_metric = -1.0
    # --- [早停] 初始化早停计数器 ---
    epochs_since_best = 0
    
    if args.resume:
        checkpoint_path = args.resume
        # 智能处理路径：如果提供的是目录，则自动查找最新的检查点
        if os.path.isdir(checkpoint_path):
            last_ckpt = Path(checkpoint_path) / "checkpoint_last.pth"
            best_ckpt = Path(checkpoint_path) / "checkpoint_best.pth"
            if last_ckpt.exists():
                checkpoint_path = str(last_ckpt)
                print(f"检测到目录，使用最新的检查点: {checkpoint_path}")
            elif best_ckpt.exists():
                checkpoint_path = str(best_ckpt)
                print(f"检测到目录，使用最佳的检查点: {checkpoint_path}")
            else:
                # 如果目录为空，则不加载任何东西，行为等同于不续训
                print(f"⚠️ 续训目录 {args.resume} 为空，将从头开始训练。")
                checkpoint_path = None 

        # 只有在确定了有效的检查点文件后，才进行加载
        if checkpoint_path and os.path.isfile(checkpoint_path):
            start_epoch, best_metric, epochs_since_best = _load_resume_checkpoint(
                model, optimizer, model_ema, checkpoint_path, args
            )
        
        # 允许命令行参数覆盖从检查点中读取的epoch
        if args.resume_epoch >= 0:
            print(f"手动覆盖起始轮次为: {args.resume_epoch}")
            start_epoch = args.resume_epoch
    # =============================

    best_val_stats = {}
    metric_name = "mAP"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 添加CSV日志文件 ==========
    import csv
    csv_path = output_dir / "training_log.csv"
    
    # 续训时以追加模式打开CSV，否则新建
    if start_epoch == 0:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_metric', 'best_metric',
                 'epoch_time', 'avg_epoch_time', 'estimated_remaining_hours',
                 'completion_time', 'grad_var'])
        print("📝 创建新的训练日志文件")
    else:
        print(f"📝 续训模式，将追加到现有日志文件: {csv_path}")
    # ====================================

    # ========== 添加时间预估代码 ==========
    import datetime
    start_time = time.time()
    epoch_times = []
    
    print(f"\n🎯 开始训练，总轮次: {args.epochs}, 起始轮次: {start_epoch}")
    # --- [早停] 打印早停状态 ---
    if args.patience > 0:
        print(f"⌛ 早停机制已启用，耐心值 (Patience) = {args.patience} 轮")
    print(f"📊 训练集 batches/epoch: {len(data_loader_train)}")
    # ====================================

    start = time.time()
    for epoch in range(start_epoch, args.epochs):
         # ====== conditional_1：动态增强强度 ======
        if args.aug_mode == "conditional_1":
            ds = getattr(data_loader_train, "dataset", None)
            if ds is not None and hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch, args.epochs)
        epoch_start = time.time()  # 记录epoch开始时间
        
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            amp=True,
            model_ema=model_ema,
        )

        if model_ema is not None:
            # 创建临时模型用于EMA评估
            eval_model = build_model(args).to(device)
            model_ema.copy_to(eval_model)
            print("📊 使用EMA模型进行评估")
        else:
            eval_model = model
            print("📊 使用原始模型进行评估")
        
        # ========== 移除单个epoch的eval CSV生成 ==========
        val_stats = _safe_evaluate(
            data_loader_val=data_loader_val,
            model_to_eval=eval_model,
            device=device,
            amp=True,
            class_names=class_names,
            threshold=args.eval_threshold,
            csv_path=None,  # 设置为None，不生成单个CSV
        )
        # ====== conditional_2：根据验证集 AP 更新弱科类 ======
        if args.aug_mode == "conditional_2":
            ds = getattr(data_loader_train, "dataset", None)
            if ds is not None and hasattr(ds, "update_hard_classes_by_ap"):
                per_class_ap = val_stats.get("per_class_ap", None)
                if per_class_ap is not None:
                    val_ap_dict = {i: float(ap) for i, ap in enumerate(per_class_ap)}
                    ds.update_hard_classes_by_ap(val_ap_dict, topk=3)

        # ========== 添加时间预估计算和打印 ==========
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # 计算平均epoch时间和剩余时间预估
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epochs - epoch - 1
        estimated_remaining = avg_epoch_time * remaining_epochs
        completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining)
        
        print(f"⏰ Epoch {epoch} 耗时: {epoch_time:.1f}s, 平均: {avg_epoch_time:.1f}s, 剩余预估: {estimated_remaining/3600:.1f}h")
        print(f"  预计完成: {completion_time.strftime('%m-%d %H:%M')}")
        # ==========================================

        primary = None
        if isinstance(val_stats, dict):
            for k in ["mAP", "map", "AP", "ap", "acc1", "acc", "top1"]:
                if k in val_stats:
                    primary = float(val_stats[k]); metric_name = k; break
        if primary is None:
            primary = -float(val_stats.get("loss", train_stats.get("loss", 0.0))) if isinstance(val_stats, dict) else -float(train_stats.get("loss", 0.0))
            metric_name = "-loss"

        is_best = primary > best_metric
        
        if is_best: 
            best_metric = primary
             # --- [早停] 如果是最佳，重置计数器 ---
            best_val_stats = val_stats 
            epochs_since_best = 0
            print(f"🎉 新的最佳性能! mAP = {best_metric:.4f}. 重置早停计数器。")
        else:
            # --- [早停] 如果不是最佳，计数器+1 ---
            epochs_since_best += 1
            print(f"📉 性能未提升，早停计数器: {epochs_since_best}/{args.patience}")


        # ========== 写入CSV日志 ==========
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_stats.get('loss', 0.0),
                primary,
                best_metric,
                epoch_time,
                avg_epoch_time,
                estimated_remaining/3600,
                completion_time.strftime('%m-%d %H:%M'),
                train_stats.get('grad_var', 0.0)
            ])
        # ================================

        ckpt = {
            "model": model.state_dict(), 
            "optimizer": optimizer.state_dict(), 
            "args": vars(args),
            "epoch": epoch, 
            "metric": {metric_name: best_metric},
            "epochs_since_best": epochs_since_best
        }
        
        # 保存EMA模型状态
        if model_ema is not None:
            ckpt["model_ema"] = model_ema.ema_state
        
        torch.save(ckpt, str(output_dir / "checkpoint_last.pth"))
        if is_best:
            torch.save(ckpt, str(output_dir / "checkpoint_best.pth"))

        took = time.time() - start
        print(f"[epoch {epoch}] val {metric_name}={primary:.4f} (best={best_metric:.4f})   elapsed={took/60.0:.1f} min")
        # --- [早停] 检查是否触发早停 ---
        if args.patience > 0 and epochs_since_best >= args.patience:
            print(f"\n🛑 触发早停! 验证集指标已连续 {args.patience} 轮未提升。")
            print(f"   - 最佳性能出现在第 {epoch - epochs_since_best} 轮，{metric_name} = {best_metric:.4f}")
            break  # 中断训练循环

    # ========== 添加训练完成总耗时 ==========
    total_time = time.time() - start_time
    print(f"\n✅ 训练完成! 总耗时: {total_time/3600:.2f} 小时")
    
    # ========== 按mAP排序CSV文件 ==========
    print("📊 按mAP排序训练日志...")
    
    try:
        df = pd.read_csv(csv_path)
        df_sorted = df.sort_values('val_metric', ascending=False)  # 按val_metric(mAP)降序排序
        df_sorted.to_csv(csv_path, index=False)
        print(f"✅ 训练日志已按mAP排序并保存至: {csv_path}")
    except Exception as e:
        print(f"⚠️ 排序CSV文件时出错: {e}")
    # =====================================
    

    # --- 【核心修改】调用新函数，并传递 per-class AP ---
    try:
        total_params = sum(p.numel() for p in model.parameters())
    except:
        total_params = 0
    
    # 从我们保存的最佳统计数据中获取 per_class_ap
    best_per_class_ap = best_val_stats.get('per_class_ap', [])

    append_summary_to_global_log(
        args=args, 
        best_metric_value=best_metric, 
        metric_name=metric_name,
        model_total_params=total_params,
        class_names=class_names, # class_names 是从 build_loaders 获取的
        per_class_ap_list=best_per_class_ap
    )
    # --------------------------------------------------

    print("Finished.")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)



