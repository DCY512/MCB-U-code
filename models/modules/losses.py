# models/modules/losses.py (完整替换)

import torch
import torch.nn as nn
from .distillation.kd_losses import LogitsDistillationLoss, DKDLoss, FeatureLoss
from .distillation.predictors import FeatureExtractor
from typing import List, Dict, Optional


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失“总调度中心” (v2, 支持特征蒸馏)。
    """
    def __init__(self, base_criterion: nn.Module, student_model: nn.Module, teacher_model: nn.Module,
                 kd_mode: str = 'logits', alpha: float = 0.5, beta: float = 1.0,
                 dkd_alpha: float = 1.0, dkd_beta: float = 8.0,
                 tau: float = 2.0, class_weights=None, 
                 feature_layers: List[str] = None, adapter_configs: Dict = None):
        super().__init__()
        self.base_criterion = base_criterion
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.kd_mode = kd_mode
        self.alpha = alpha # 平衡因子 (硬损失)
        self.beta = beta   # 平衡因子 (特征损失)

        # --- Logits / DKD 蒸馏器 ---
        if kd_mode == 'logits':
            self.logits_distiller = LogitsDistillationLoss(tau=tau, class_weights=class_weights)
        elif kd_mode == 'dkd':
            self.logits_distiller = DKDLoss(alpha=dkd_alpha, beta=dkd_beta, temperature=tau)
        else:
            self.logits_distiller = None
        
        # --- 特征蒸馏器 ---
        self.feature_distiller = None
        if feature_layers:
            print("🔥 特征蒸馏已启用!")
            student_extractor = FeatureExtractor(student_model.backbone, feature_layers)
            teacher_extractor = FeatureExtractor(teacher_model.backbone, feature_layers)
            self.feature_distiller = FeatureLoss(student_extractor, teacher_extractor, adapter_configs)
            
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def forward(self, student_outputs, student_inputs, targets):
        xa, xb = student_inputs
        
        hard_loss = self.base_criterion(student_outputs, targets)
        
        soft_loss = 0.0
        if self.logits_distiller:
            with torch.no_grad():
                teacher_logits = self.teacher_model(xa, xb) if xb is not None else self.teacher_model(xa)
                if isinstance(teacher_logits, dict): teacher_logits = teacher_logits['logits']

            if self.kd_mode == 'dkd':
                soft_loss = self.logits_distiller(student_outputs, teacher_logits, targets)
            else: # 'logits'
                soft_loss = self.logits_distiller(student_outputs, teacher_logits)
        
        feature_loss = 0.0
        if self.feature_distiller:
            # 假设双视角蒸馏时，只使用第一个视角进行特征匹配
            feature_loss = self.feature_distiller(xa, xa)

        # 最终损失组合: Hard + Soft + Feature
        # alpha 控制硬损失权重，beta 控制特征损失权重，软损失权重由 1-alpha-beta 动态决定
        soft_weight = max(1.0 - self.alpha - self.beta, 0.0)
        
        total_loss = self.alpha * hard_loss + soft_weight * soft_loss + self.beta * feature_loss
        return total_loss