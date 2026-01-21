# models/modules/necks.py (已修正的最终版本)
from typing import Dict, List, Union, Optional
import torch.nn as nn
import torch.nn.functional as F

# 从同级目录的 attentions.py 中导入注册表
from .attentions import ATTENTION_REGISTRY

# --- 新增: 并联注意力模块 ---
class ParallelAttentionModule(nn.Module):
    """
    接收多个注意力模块，并联处理输入后将结果相加。
    """
    def __init__(self, attn_modules: List[nn.Module]):
        super().__init__()
        self.attns = nn.ModuleList(attn_modules)
    
    def forward(self, x):
        outputs = [attn(x) for attn in self.attns]
        return sum(outputs)
# ==============================
# FPN + PAN 模块 (已集成注意力)
# ==============================
class FPN_PAN(nn.Module):
    def __init__(
        self, c3, c4, c5, out_channels=256, 
        attention_config: Optional[Dict] = None
    ):
        super().__init__()
        

        # FPN Layers
        self.fpn_l3 = nn.Conv2d(c3, out_channels, 1)
        self.fpn_s3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_l4 = nn.Conv2d(c4, out_channels, 1)
        self.fpn_s4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_l5 = nn.Conv2d(c5, out_channels, 1)
        self.fpn_s5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # PAN Layers
        self.pan_d4 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.pan_d5 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.pan_s3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pan_s4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pan_s5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
         
        # --- 【核心改造区】---
        self.attn3, self.attn4, self.attn5 = None, None, None
        if attention_config:
            if "N3" in attention_config:
                self.attn3 = self._build_attention_block(attention_config["N3"], out_channels)
            if "N4" in attention_config:
                self.attn4 = self._build_attention_block(attention_config["N4"], out_channels)
            if "N5" in attention_config:
                self.attn5 = self._build_attention_block(attention_config["N5"], out_channels)

    def _build_attention_block(self, config: Union[str, list, dict], channels: int) -> nn.Module:
        if isinstance(config, str):
            attn_class = ATTENTION_REGISTRY.get(config.lower())
            if not attn_class: raise ValueError(f"未知的注意力类型: {config}")
        
            if config.lower() == "coordatt":
                 return attn_class(channels, channels) # in_planes, out_planes
            if config.lower() == "external":
                 return attn_class(channels, channels // 2) # in_channels, inter_channels

            # 对于其他常规注意力模块，使用此默认方式
            return attn_class(channels)

        elif isinstance(config, list):
            modules = [self._build_attention_block(name, channels) for name in config]
            return nn.Sequential(*modules)
            
        elif isinstance(config, dict) and 'parallel' in config:
            modules = [self._build_attention_block(name, channels) for name in config['parallel']]
            return ParallelAttentionModule(modules)
            
        else:
            raise TypeError(f"不支持的注意力配置格式: {config}")

    def forward(self, C3, C4, C5):
        # Top-Down FPN Path
        P5 = self.fpn_s5(self.fpn_l5(C5))
        P4 = self.fpn_s4(self.fpn_l4(C4) + F.interpolate(P5, size=C4.shape[-2:], mode='nearest'))
        P3 = self.fpn_s3(self.fpn_l3(C3) + F.interpolate(P4, size=C3.shape[-2:], mode='nearest'))

        # Bottom-Up PAN Path
        N3 = self.pan_s3(P3)
        N4 = self.pan_s4(P4 + self.pan_d4(N3))
        N5 = self.pan_s5(P5 + self.pan_d5(N4))

        # --- 【核心改造区】---
        # 按需、独立地应用注意力
        if self.attn3: N3 = self.attn3(N3)
        if self.attn4: N4 = self.attn4(N4)
        if self.attn5: N5 = self.attn5(N5)
            
        return N3, N4, N5

# ------------------------------
# FPN 模块（直接集成到这里）
# ------------------------------
class FPN(nn.Module):
    """
    最小FPN：输入 C3/C4/C5，输出 P3/P4/P5
    """
    def __init__(self, c3, c4, c5, out_channels=256):
        super().__init__()
        # 侧连 1x1
        self.l3 = nn.Conv2d(c3, out_channels, 1)
        self.l4 = nn.Conv2d(c4, out_channels, 1)
        self.l5 = nn.Conv2d(c5, out_channels, 1)
        # 平滑 3x3
        self.s3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.s4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.s5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, C3, C4, C5):
        P5 = self.l5(C5)
        P4 = self.l4(C4) + F.interpolate(P5, size=C4.shape[-2:], mode='nearest')
        P3 = self.l3(C3) + F.interpolate(P4, size=C3.shape[-2:], mode='nearest')
        return self.s3(P3), self.s4(P4), self.s5(P5)  # P3,P4,P5

