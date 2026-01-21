# -*- coding: utf-8 -*-
"""
DVXRAY 双视角数据集加载器（datasets.py）
- 返回 ((imgA, imgB), target)
- 训练/验证两种模式（同步增广）
- 通过 build_loaders(args) 构建 DataLoader
"""

import os
import random
from typing import List, Tuple
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from models.modules.augmentations import RandAugment, TrivialAugmentWide


# ===========================
# 基本常量
# ===========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ===========================
# 辅助函数
# ===========================
def _read_class_names(path: str, num_classes: int) -> List[str]:
    """读取类别名文件；若无则生成 cls_0, cls_1 ..."""
    if path and os.path.isfile(path):
        names = [ln.strip() for ln in open(path, "r", encoding="utf-8") if ln.strip()]
        if len(names) >= num_classes:
            return names[:num_classes]
    return [f"cls_{i}" for i in range(num_classes)]

def _parse_labels(tokens: List[str], num_classes: int) -> torch.Tensor:
    """解析多标签：支持以空格或逗号分隔"""
    if len(tokens) == 1 and ("," in tokens[0]):
        vals = [int(x) for x in tokens[0].replace(",", " ").split()]
    else:
        vals = [int(x) for x in tokens]
    if len(vals) != num_classes:
        raise ValueError(f"标签维度不对：期望 {num_classes}，实际 {len(vals)}；内容={vals[:10]}")
    return torch.tensor(vals, dtype=torch.float32)

def _parse_line(line: str, num_classes: int) -> Tuple[str, str, torch.Tensor]:
    """
    每行格式：
        imgA_path imgB_path label1 label2 ... labelK
    或
        imgA_path imgB_path 1,0,0,1,...,0
    """
    toks = line.strip().split()
    if len(toks) < 3:
        raise ValueError(f"标注行错误：{line}")
    a, b = toks[0], toks[1]
    y = _parse_labels(toks[2:], num_classes)
    return a, b, y

# ===========================
# 【核心修改】同步数据增强类
# ===========================
class SynchronizedTransform:
    """
    一个封装类，确保对双视角的两个图像应用完全相同的随机参数。
    """
    def __init__(self, transform: nn.Module):
        self.transform = transform

    def __call__(self, imgA: Image.Image, imgB: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # 保存并设置随机状态，确保两次调用transform时随机参数一致
        seed = random.randint(0, 2**32)
        random.seed(seed)
        torch.manual_seed(seed)
        imgA = self.transform(imgA)
        
        random.seed(seed)
        torch.manual_seed(seed)
        imgB = self.transform(imgB)
        
        return imgA, imgB


# ===========================
# 数据集类 (最终修正版)
# ===========================
class DualViewTxtDataset(Dataset):
    def __init__(self, list_file: str, input_size: int, num_classes: int, train: bool, class_names: List[str], aug_mode: str = 'standard', rand_aug_n: int = 2, rand_aug_m: int = 9):
        super().__init__()
        self.samples = []
        with open(list_file, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    self.samples.append(ln.strip())

        self.num_classes = num_classes
        self.input_size = input_size
        self.train = train
        self.use_conditional_aug = (aug_mode == 'conditional')

        # --- 【修正】识别“弱科”类别 ---
        hard_class_names = {"Scissors", "Lighter", "Razor_blade","Knife"}
        self.hard_class_indices = {i for i, name in enumerate(class_names) if name in hard_class_names}
        self.class_names = class_names
        self.class_to_idx = {n: i for i, n in enumerate(self.class_names)}

        if self.train:
            # 只有在“总开关”开启时，才打印“已启用”的日志
            if self.use_conditional_aug:
                print(f"✅ [专项增强] 已启用 (ON)，将对弱科类别索引 {self.hard_class_indices} 应用强化训练。")
            else:
                print("ℹ️  [专项增强] 已关闭 (OFF)，所有样本将使用标准数据增强。")

        # --- 【修正】用真实的增强流程替换占位符 ---
        if self.train:
            # 准备基础增强 (裁剪和翻转)
            base_transforms = [
                T.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(3./4., 4./3.)),
                T.RandomHorizontalFlip(),
            ]
            if aug_mode == 'none':
                # 【新】模式0: 无增强模式 (只做 Resize)
                self.transforms = SynchronizedTransform(T.Resize((input_size, input_size), antialias=True)) 

            elif aug_mode == 'standard':
                # 模式1: 标准模式 (只加入温和的颜色抖动)
                self.transforms = SynchronizedTransform(T.Compose(base_transforms + [T.ColorJitter(0.1, 0.1, 0.1, 0.05)]))
            
            elif aug_mode == 'conditional':
                # 模式2: 条件模式 (准备两套)
                self.normal_transforms = SynchronizedTransform(T.Compose(base_transforms + [T.ColorJitter(0.1, 0.1, 0.1, 0.05)]))
                self.strong_transforms = SynchronizedTransform(T.Compose([
                    T.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(15),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]))

            elif aug_mode == 'conditional_1':
                # 渐进式增强：随 epoch 逐步增强裁剪强度（需要引擎每轮调用 set_epoch）
                self.aug_mode = 'conditional_1'
                self.total_epochs = None   # 由引擎设定
                self.current_epoch = 0
                # 常规（较弱）增强
                self.normal_transforms = SynchronizedTransform(T.Compose([
                    T.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(3./4., 4./3.)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                ]))
                # 占位；根据 epoch 动态重建
                self.progressive_transforms = None

            elif aug_mode == 'conditional_2':
                # 类别难度差异化增强：对低 AP 的“弱科类”样本施加强增强
                self.aug_mode = 'conditional_2'
                self.normal_transforms = SynchronizedTransform(T.Compose([
                    T.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(3./4., 4./3.)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                ]))
                self.strong_transforms = SynchronizedTransform(T.Compose([
                    T.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(15),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]))
                # 默认沿用你原来的启发式“弱科”，后续由引擎的验证 AP 更新
                self.hard_class_indices = set(self.hard_class_indices)
            elif aug_mode == 'conditional_3':
                self.aug_mode = 'conditional_3'
                self.normal_transforms = SynchronizedTransform(T.Compose([
                    T.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(3./4., 4./3.)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                ]))
                self.strong_transforms = SynchronizedTransform(T.Compose([
                    T.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(15),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]))
                hard_class_names = {"Lighter", "Razor_blade"}
                self.hard_class_indices = {
                    self.class_to_idx[n] for n in hard_class_names if n in self.class_to_idx
                }

            elif aug_mode == 'conditional_4':
                self.aug_mode = 'conditional_4'
                self.normal_transforms = SynchronizedTransform(T.Compose([
                    T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                ]))
                self.strong_transforms = SynchronizedTransform(T.Compose([
                    T.RandomResizedCrop(input_size, scale=(0.7, 1.0)),   # 比 conditional_2 温和
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(10),
                    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                ]))
                # 沿用已有弱科集合（可在 __init__ 顶部的 hard_class_names 调整）
                self.hard_class_indices = set(getattr(self, "hard_class_indices", set()))



            elif aug_mode == 'rand_aug':
                # 模式3: RandAugment 模式
                self.transforms = SynchronizedTransform(T.Compose(base_transforms + [RandAugment(n=rand_aug_n, m=rand_aug_m)]))
            
            elif aug_mode == 'trivial_aug':
                # 模式4: TrivialAugment 模式
                self.transforms = SynchronizedTransform(T.Compose(base_transforms + [TrivialAugmentWide()]))
            
            else:
                raise ValueError(f"Unknown aug_mode: {aug_mode}")
        
        else: # 验证集
            self.val_transforms = SynchronizedTransform(T.Resize((input_size, input_size), antialias=True))


        # 最终转换为Tensor和归一化的操作是共用的
        self.to_tensor_and_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.samples)

    # --- 【修正】旧的、冗余的数据增强函数已被安全删除 ---

    def __getitem__(self, idx: int):
        a_path, b_path, y = _parse_line(self.samples[idx], self.num_classes)
        if not os.path.isfile(a_path) or not os.path.isfile(b_path):
            raise FileNotFoundError(f"缺少图像：{a_path} 或 {b_path}")

        imgA = Image.open(a_path).convert("RGB")
        imgB = Image.open(b_path).convert("RGB")
        
        if self.train:
            if getattr(self, "aug_mode", None) == 'conditional_1':
                # 渐进式：始终使用 progressive_transforms（随 epoch 已更新）
                if self.progressive_transforms is None:
                    # 首轮兜底：相当于 normal
                    imgA, imgB = self.normal_transforms(imgA, imgB)
                else:
                    imgA, imgB = self.progressive_transforms(imgA, imgB)

            elif getattr(self, "aug_mode", None) in ('conditional_2', 'conditional_3', 'conditional_4') or self.use_conditional_aug:
                # 条件增强：按“弱科类”样本分流（conditional_2/3/4 以及旧版 conditional 共用这段）
                contains_hard_class = any((y[i] == 1) for i in self.hard_class_indices)
                if contains_hard_class and self.strong_transforms is not None:
                    imgA, imgB = self.strong_transforms(imgA, imgB)
                else:
                    # 旧版 conditional 走 normal_transforms；standard/none 则走 self.transforms
                    if hasattr(self, "normal_transforms") and self.normal_transforms is not None:
                        imgA, imgB = self.normal_transforms(imgA, imgB)
                    else:
                        imgA, imgB = self.transforms(imgA, imgB)

            else:
                # 其他训练模式：standard / none / rand_aug / trivial_aug
                imgA, imgB = self.transforms(imgA, imgB)

        else:
            # 验证集统一走 val_transforms
            imgA, imgB = self.val_transforms(imgA, imgB)

        # 最后的 toTensor 和归一化
        ta = self.to_tensor_and_norm(imgA)
        tb = self.to_tensor_and_norm(imgB)
        
        return (ta, tb), y

    # 用于 conditional_1：由引擎每轮设置 epoch，并据此重建 progressive_transforms
    def set_epoch(self, epoch: int, total_epochs: int = None):
        if not self.train: 
            return
        if getattr(self, "aug_mode", None) != 'conditional_1':
            return
        if total_epochs is not None:
            self.total_epochs = total_epochs
        self.current_epoch = int(epoch)
        te = float(self.total_epochs or 1)
        # 你的公式：scale_min = max(0.6, 0.6 + 0.1*(epoch/total_epochs))
        scale_min = max(0.6, 0.6 + 0.1 * (self.current_epoch / max(te, 1.0)))
        self.progressive_transforms = SynchronizedTransform(T.Compose([
            T.RandomResizedCrop(self.input_size, scale=(scale_min, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.15, 0.15, 0.15, 0.08),
        ]))

    # 用于 conditional_2：由引擎在每轮验证后更新“弱科”类集合
    def update_hard_classes_by_ap(self, val_ap_dict: dict, topk: int = 3):
        """
        val_ap_dict: {class_index(int) or class_name(str): ap_value(float)}
        低 AP 的前 topk 类将作为“弱科类”
        """
        if not self.train:
            return
        if getattr(self, "aug_mode", None) != 'conditional_2':
            return

        # 兼容 class_name 与 index 两种键
        items = []
        for k, v in val_ap_dict.items():
            if isinstance(k, str):
                # 需要从 name 找 index；若失败则跳过
                try:
                    idx = int(k) if k.isdigit() else None
                except:
                    idx = None
                if idx is None:
                    # 退化：按名字匹配 class_names
                    # 这里 datasets.py 里没有 class_names 保存，如需名字→idx，请在 __init__ 存起来
                    pass
            else:
                idx = int(k)
            if 0 <= idx < self.num_classes:
                items.append((idx, float(v)))
        if not items:
            return
        items.sort(key=lambda x: x[1])  # 按 AP 升序
        new_hard = {idx for idx, _ in items[:max(1, topk)]}
        self.hard_class_indices = set(new_hard)
        print(f"✅ [conditional_2] 更新弱科类索引: {sorted(self.hard_class_indices)}")
# ===========================
# DataLoader 构建函数
# ===========================
def build_loaders(args):
    """
    供 main_finetune.py 调用
    args 需包含：
        train_list / val_list / classes_file / input_size / batch_size / num_classes
    返回：
        train_loader, val_loader, class_names
    """
    num_workers = getattr(args, "num_workers", 8)
    class_names = _read_class_names(
        getattr(args, "classes_file", ""), args.num_classes
    )
    # --- 【核心修正】从 args 中读取所有新的增强参数 ---
    train_set = DualViewTxtDataset(
        args.train_list, args.input_size, args.num_classes, train=True, class_names=class_names,
        aug_mode=args.aug_mode,
        rand_aug_n=args.rand_aug_n,
        rand_aug_m=args.rand_aug_m
    )
    # -----------------------------------------------

    val_set = DualViewTxtDataset(
        args.val_list, args.input_size, args.num_classes, train=False, class_names=class_names
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, class_names
