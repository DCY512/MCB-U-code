# models/modules/composite_loss.py
import torch
import torch.nn as nn
from .custom_losses.focal_loss import FocalLoss
from .custom_losses.fals_loss import FALSLoss
from .custom_losses.mcb_loss import MCBLoss, MCBLossConvex
from .custom_losses.gebce import GEBCELoss
from .custom_losses.dals_loss import DALSBCE as DALSLoss
from .custom_losses.consistency_losses import (
    UncertaintyConsistencyLoss, ChannelAttentionConsistencyLoss, RelationalConsistencyLoss
)


class CompositeLoss(nn.Module):
    """
    一个统一的、智能的复合损失函数，用于“正常训练”模式。
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.base_loss == 'mcb_convex':
            self.base_criterion = MCBLossConvex(tau=args.mcb_tau, w_min=args.mcb_wmin, momentum=args.mcb_momentum)
        elif args.base_loss == 'gebce': 
            self.base_criterion = GEBCELoss(lambda_coef=args.ge_lambda, pos_only=args.ge_pos_only,
                                alpha=args.ge_alpha, ema=args.ge_ema,
                                momentum=args.ge_momentum, band=args.ge_band,
                                trainable=args.ge_trainable)
        elif args.base_loss == 'dals':
            self.base_criterion = DALSLoss(eps=args.dals_eps, gamma=args.dals_gamma)
        # 1. 初始化基础监督损失
        elif args.base_loss == 'focal':
            self.base_criterion = FocalLoss(gamma=args.focal_gamma)
        elif args.base_loss == 'fals':
            self.base_criterion = FALSLoss(eps=args.fals_eps, gamma=args.fals_gamma)
        elif args.base_loss == 'mcb':
            self.base_criterion = MCBLoss(tau=args.mcb_tau, momentum=args.mcb_momentum)
        elif args.base_loss == 'mlsm':
            self.base_criterion = nn.MultiLabelSoftMarginLoss()
        else:  # 'bce'
            self.base_criterion = nn.BCEWithLogitsLoss()

        # 2. 初始化所有可选的一致性损失
        self.uncert_loss = UncertaintyConsistencyLoss() if args.use_uncertainty_loss else None
        self.chan_loss = ChannelAttentionConsistencyLoss() if args.use_channel_loss else None
        self.rel_loss = RelationalConsistencyLoss() if args.use_relational_loss else None

    def forward(self, model_outputs, targets):
        # model_outputs 是您的 ConvNeXtV2Dual.forward 的原始返回
        # e.g., {"logits": ..., "feats": {"A": ..., "B": ..., "fused": ...}}

        final_logits = model_outputs['logits']
        
        # --- 1. 计算主损失 ---
        total_loss = self.base_criterion(final_logits, targets)

        # --- 2. 按需计算并累加所有辅助损失 ---
        # 为了计算辅助损失，我们需要从模型输出中提取各种中间结果
        if self.uncert_loss and all(k in model_outputs for k in ['A_logits','B_logits']):
            total_loss += self.args.uncertainty_lambda * self.uncert_loss(
                model_outputs['A_logits'], model_outputs['B_logits'])

        if self.uncert_loss:
            total_loss += self.args.uncertainty_lambda * self.uncert_loss(logits_a, logits_b)
        
        # ... (类似地为 channel_loss 和 relational_loss 添加计算) ...
        
        return total_loss