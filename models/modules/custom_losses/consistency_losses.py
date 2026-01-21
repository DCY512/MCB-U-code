# models/modules/custom_losses/consistency_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyConsistencyLoss(nn.Module):
    def forward(self, logits_a, logits_b):
        p_a = F.softmax(logits_a, dim=1)
        p_b = F.softmax(logits_b, dim=1)
        
        h_a = torch.sum(-p_a * torch.log(p_a + 1e-8), dim=1)
        h_b = torch.sum(-p_b * torch.log(p_b + 1e-8), dim=1)
        
        return F.l1_loss(h_a, h_b)

class ChannelAttentionConsistencyLoss(nn.Module):
    def forward(self, feats_a, feats_b):
        v_a = F.adaptive_avg_pool2d(feats_a, 1).squeeze()
        v_b = F.adaptive_avg_pool2d(feats_b, 1).squeeze()
        
        return 1 - F.cosine_similarity(v_a, v_b, dim=-1).mean()

class RelationalConsistencyLoss(nn.Module):
    def forward(self, x_a, x_b): # x_a, x_b are GAP features [B, D]
        x_a = F.normalize(x_a, p=2, dim=1)
        x_b = F.normalize(x_b, p=2, dim=1)
        
        m_a = x_a @ x_a.T
        m_b = x_b @ x_b.T
        
        return F.mse_loss(m_a, m_b)