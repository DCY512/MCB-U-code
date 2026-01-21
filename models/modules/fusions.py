from typing import Tuple
import torch
import torch.nn as nn
from typing import Tuple, Dict
# 从同级目录的 common.py 中导入 Conv1x1
from .common import Conv1x1
import torch.nn.functional as F 

# ------------------------------
# Gated 融合：concat → 1x1 降维 → SE 门控
# ------------------------------
class GatedFuse(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, r: int = 8):
        super().__init__()
        hid = max(out_ch // r, 1)
        self.reduce = Conv1x1(in_ch, out_ch, act=True)   # 2C -> C
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_ch, hid, 1, bias=True)
        self.fc2 = nn.Conv2d(hid, out_ch, 1, bias=True)
        self.act = nn.GELU()
        self.sig = nn.Sigmoid()

    def forward(self, xa, xb):
        x = torch.cat([xa, xb], dim=1)      # [B,2C,H,W]
        y = self.reduce(x)                  # [B,C,H,W]
        w = self.avg(y)                     # [B,C,1,1]
        w = self.fc2(self.act(self.fc1(w)))
        w = self.sig(w)
        return y * w + y * (1 - w) * 0.0    # 保留结构，等价于 y * w


# ------------------------------
# XAttn 融合：双向跨视角注意力（可下采样降低 token 数）
# A 作为 Query、B 作为 KV 得到 A←B 的信息；反向亦然，然后平均融合 + 残差回传
# ------------------------------
class XAttnFuse(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, reduction: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels 必须能被 num_heads 整除"
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.reduction = max(int(reduction), 1)

        # PyTorch MultiheadAttention 期望 [S,B,E]
        self.mha_ab = nn.MultiheadAttention(embed_dim=self.channels, num_heads=self.num_heads, batch_first=False)
        self.mha_ba = nn.MultiheadAttention(embed_dim=self.channels, num_heads=self.num_heads, batch_first=False)

        # 归一化 & 输出映射
        self.norm_q = nn.LayerNorm(self.channels)
        self.norm_kv = nn.LayerNorm(self.channels)
        self.proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=self.reduction, stride=self.reduction, ceil_mode=False)

    def _upsample(self, x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        if self.reduction <= 1:
            return x
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

    def _to_seq(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [S(=H*W), B, C] with LN for Q
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)      # [B, HW, C]
        x = self.norm_q(x)
        x = x.transpose(0, 1)                 # [HW, B, C]
        return x, (H, W)

    def _to_seq_kv(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)      # [B, HW, C]
        x = self.norm_kv(x)
        x = x.transpose(0, 1)                 # [HW, B, C]
        return x

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = xa.shape

        # 空间下采样，降低 token 数（更省显存）
        xa_ds = self._downsample(xa)
        xb_ds = self._downsample(xb)
        h, w = xa_ds.shape[-2], xa_ds.shape[-1]

        # token 化
        qa, _ = self._to_seq(xa_ds)       # [S,B,C]
        kab = self._to_seq_kv(xb_ds)      # [S,B,C]
        qb, _ = self._to_seq(xb_ds)
        kba = self._to_seq_kv(xa_ds)

        # 双向 cross-attn
        y_ab, _ = self.mha_ab(qa, kab, kab, need_weights=False)  # A <- B
        y_ba, _ = self.mha_ba(qb, kba, kba, need_weights=False)  # B <- A

        # 回到 [B,C,h,w]
        y_ab = y_ab.transpose(0, 1).transpose(1, 2).reshape(B, C, h, w)
        y_ba = y_ba.transpose(0, 1).transpose(1, 2).reshape(B, C, h, w)

        # 上采样回原尺寸
        y_ab = self._upsample(y_ab, (H, W))
        y_ba = self._upsample(y_ba, (H, W))

        # 融合：平均后再 1x1，并加入残差（更稳）
        y = 0.5 * (y_ab + y_ba)
        y = self.proj(y)
        return xa + 0.5 * y   # 残差：对原特征做温和调
    
# ==============================
# [新增] AHCR 核心组件：自适应交叉注意力
# ==============================
# ==============================
# [修正] AHCR 核心组件：自适应交叉注意力 (完全版)
# ==============================
class AdaptiveCrossAttention(nn.Module):
    def __init__(self, dim_query, dim_kv, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_query // num_heads # Head dimension is based on query dim
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim_query, dim_query, bias=qkv_bias)
        self.wk = nn.Linear(dim_kv, dim_query, bias=qkv_bias)
        self.wv = nn.Linear(dim_kv, dim_query, bias=qkv_bias)
        
        self.proj = nn.Linear(dim_query, dim_query)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_query, x_key_value):
        B, N_q, C_q = x_query.shape
        _, N_kv, _ = x_key_value.shape
        
        q = self.wq(x_query).reshape(B, N_q, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x_key_value).reshape(B, N_kv, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x_key_value).reshape(B, N_kv, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C_q)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x_query + self.gamma * x

# ==============================
# [修正] AHCR 融合模块 (终极版)
# ==============================
class AHCRFuse(nn.Module):
    def __init__(self, c3_dim, c4_dim, c5_dim, mode='intra_level'):
        super().__init__()
        self.mode = mode

        if self.mode == 'intra_level':
            # 在层内模式下，query 和 kv 维度相同
            self.cross_attn_c3 = AdaptiveCrossAttention(dim_query=c3_dim, dim_kv=c3_dim)
            self.cross_attn_c3_rev = AdaptiveCrossAttention(dim_query=c3_dim, dim_kv=c3_dim)
            self.cross_attn_c4 = AdaptiveCrossAttention(dim_query=c4_dim, dim_kv=c4_dim)
            self.cross_attn_c4_rev = AdaptiveCrossAttention(dim_query=c4_dim, dim_kv=c4_dim)
            self.cross_attn_c5 = AdaptiveCrossAttention(dim_query=c5_dim, dim_kv=c5_dim)
            self.cross_attn_c5_rev = AdaptiveCrossAttention(dim_query=c5_dim, dim_kv=c5_dim)
            
            self.final_fuse_c3 = Conv1x1(in_ch=2*c3_dim, out_ch=c3_dim, act=True)
            self.final_fuse_c4 = Conv1x1(in_ch=2*c4_dim, out_ch=c4_dim, act=True)
            self.final_fuse_c5 = Conv1x1(in_ch=2*c5_dim, out_ch=c5_dim, act=True)
        
        elif self.mode == 'inter_level':
            self.cross_attn_c4_to_c3 = AdaptiveCrossAttention(dim_query=c3_dim, dim_kv=c4_dim)
            self.cross_attn_c4_to_c3_rev = AdaptiveCrossAttention(dim_query=c3_dim, dim_kv=c4_dim)
            self.cross_attn_c5_to_c4 = AdaptiveCrossAttention(dim_query=c4_dim, dim_kv=c5_dim)
            self.cross_attn_c5_to_c4_rev = AdaptiveCrossAttention(dim_query=c4_dim, dim_kv=c5_dim)
            
            self.final_fuse_c3 = Conv1x1(in_ch=2*c3_dim, out_ch=c3_dim, act=True)
            self.final_fuse_c4 = Conv1x1(in_ch=2*c4_dim, out_ch=c4_dim, act=True)
            self.final_fuse_c5 = nn.Identity()
        else:
            raise ValueError(f"未知的 AHCR 模式: {self.mode}")

    def _flatten(self, x):
        return x.flatten(2).transpose(1, 2)

    def _reshape(self, x, B, C, H, W):
        return x.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, feats_a: Dict[str, torch.Tensor], feats_b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c3a, c4a, c5a = feats_a['C3'], feats_a['C4'], feats_a['C5']
        c3b, c4b, c5b = feats_b['C3'], feats_b['C4'], feats_b['C5']
        
        B, C3, H3, W3 = c3a.shape
        _, C4, H4, W4 = c4a.shape
        _, C5, H5, W5 = c5a.shape
        
        if self.mode == 'intra_level':
            c3a_r = self.cross_attn_c3(self._flatten(c3a), self._flatten(c3b))
            c3b_r = self.cross_attn_c3_rev(self._flatten(c3b), self._flatten(c3a))
            fused_c3 = self.final_fuse_c3(torch.cat([
                self._reshape(c3a_r, B, C3, H3, W3),
                self._reshape(c3b_r, B, C3, H3, W3)
            ], dim=1))

            c4a_r = self.cross_attn_c4(self._flatten(c4a), self._flatten(c4b))
            c4b_r = self.cross_attn_c4_rev(self._flatten(c4b), self._flatten(c4a))
            fused_c4 = self.final_fuse_c4(torch.cat([
                self._reshape(c4a_r, B, C4, H4, W4),
                self._reshape(c4b_r, B, C4, H4, W4)
            ], dim=1))

            c5a_r = self.cross_attn_c5(self._flatten(c5a), self._flatten(c5b))
            c5b_r = self.cross_attn_c5_rev(self._flatten(c5b), self._flatten(c5a))
            fused_c5 = self.final_fuse_c5(torch.cat([
                self._reshape(c5a_r, B, C5, H5, W5),
                self._reshape(c5b_r, B, C5, H5, W5)
            ], dim=1))
            
            return {'C3': fused_c3, 'C4': fused_c4, 'C5': fused_c5}

        elif self.mode == 'inter_level':
            # --- 跨层交叉逻辑 (示例实现) ---
            # 精炼 C3a (细) & C3b (细): 分别用 c4b (粗) 和 c4a (粗) 作为 key/value
            c4b_resized = F.interpolate(c4b, size=(H3, W3), mode='bilinear', align_corners=False)
            c3a_r = self.cross_attn_c4_to_c3(self._flatten(c3a), self._flatten(c4b_resized))
            
            c4a_resized = F.interpolate(c4a, size=(H3, W3), mode='bilinear', align_corners=False)
            c3b_r = self.cross_attn_c4_to_c3_rev(self._flatten(c3b), self._flatten(c4a_resized))

            fused_c3 = self.final_fuse_c3(torch.cat([
                self._reshape(c3a_r, B, C3, H3, W3),
                self._reshape(c3b_r, B, C3, H3, W3)
            ], dim=1))

            # 精炼 C4a (中) & C4b (中): 分别用 c5b (最粗) 和 c5a (最粗) 作为 key/value
            c5b_resized = F.interpolate(c5b, size=(H4, W4), mode='bilinear', align_corners=False)
            c4a_r = self.cross_attn_c5_to_c4(self._flatten(c4a), self._flatten(c5b_resized))
            
            c5a_resized = F.interpolate(c5a, size=(H4, W4), mode='bilinear', align_corners=False)
            c4b_r = self.cross_attn_c5_to_c4_rev(self._flatten(c4b), self._flatten(c5a_resized))

            fused_c4 = self.final_fuse_c4(torch.cat([
                self._reshape(c4a_r, B, C4, H4, W4),
                self._reshape(c4b_r, B, C4, H4, W4)
            ], dim=1))

            # C5 层级最高，没有更粗的层级来精炼它，所以直接用 A 视角的 (或简单融合)
            fused_c5 = self.final_fuse_c5(c5a)
            return {'C3': fused_c3, 'C4': fused_c4, 'C5': fused_c5}



try:
    from .custom_losses.signals import signal_bus
except Exception:
    signal_bus = None

class DVDisagreeGate(nn.Module):
    """
    双视角分歧门：对特征施加 (1 + λ * disagree) 的样本级增益。
    为最小侵入式实现（标量门），先保证稳定；后续可扩展为空间图门控。
    """
    def __init__(self, lam: float = 0.5):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        if signal_bus is None:
            return x
        s = signal_bus.get()["dv_disagree"]
        if s is None:
            return x
        gain = (1.0 + self.lam * s).to(dtype=x.dtype, device=x.device)
        return x * gain