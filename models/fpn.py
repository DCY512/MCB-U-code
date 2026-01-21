# models/fpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    """
    最小FPN：输入 C3/C4/C5，输出 P3/P4/P5
    c3,c4,c5 通道数来自 backbone.dims[1:4]（ConvNeXt{,V2} 常见）
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
