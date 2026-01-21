# models/modules/distillation/kd_losses.py

import torch.nn as nn
import torch.nn.functional as F
from .utils import dkd_loss
from typing import Dict
class LogitsDistillationLoss(nn.Module):
    """ 经典 Logits 蒸馏损失 (KL 散度) """
    def __init__(self, tau=2.0, class_weights=None):
        super().__init__()
        self.tau = tau
        self.class_weights = class_weights

    def forward(self, student_logits, teacher_logits):
        reduction = 'none' if self.class_weights is not None else 'batchmean'
        
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.tau, dim=1),
            F.softmax(teacher_logits / self.tau, dim=1),
            reduction=reduction
        ) * (self.tau ** 2)

        if self.class_weights is not None:
            soft_loss = (soft_loss * self.class_weights).mean()
            
        return soft_loss

class DKDLoss(nn.Module):
    """ DKD 蒸馏损失 """
    def __init__(self, alpha=1.0, beta=8.0, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
    
    def forward(self, student_logits, teacher_logits, target):
        return dkd_loss(student_logits, teacher_logits, target, self.alpha, self.beta, self.temperature)


# models/modules/distillation/kd_losses.py (在文件末尾追加)

class FeatureLoss(nn.Module):
    """
    最基础的特征蒸馏损失 (L2 损失)。
    """
    def __init__(self, student_feature_extractor, teacher_feature_extractor, 
                 adapter_configs: Dict[str, Dict[int, int]] = None):
        """
        Args:
            student_feature_extractor: 学生的特征提取器。
            teacher_feature_extractor: 教师的特征提取器。
            adapter_configs: 一个字典，用于定义适配器层。
                             e.g., {'stages.0': {'in_dim': 96, 'out_dim': 128}, ...}
        """
        super().__init__()
        self.student_extractor = student_feature_extractor
        self.teacher_extractor = teacher_feature_extractor
        self.loss_fn = nn.MSELoss()
        
        self.adapters = nn.ModuleDict()
        if adapter_configs:
            for layer_name, dims in adapter_configs.items():
                adapter = nn.Conv2d(dims['in_dim'], dims['out_dim'], 1)
                self.adapters[layer_name.replace('.', '_')] = adapter

    def forward(self, student_inputs, teacher_inputs):
        student_features = self.student_extractor(student_inputs)
        
        with torch.no_grad():
            teacher_features = self.teacher_extractor(teacher_inputs)
            
        total_loss = 0.0
        
        for layer_name in teacher_features:
            s_feat = student_features[layer_name]
            t_feat = teacher_features[layer_name]

            # 如果定义了适配器，则使用它来对齐学生特征
            adapter_key = layer_name.replace('.', '_')
            if adapter_key in self.adapters:
                s_feat = self.adapters[adapter_key](s_feat)
            
            total_loss += self.loss_fn(s_feat, t_feat)
            
        return total_loss