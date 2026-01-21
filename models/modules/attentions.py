import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from .custom_losses.signals import signal_bus
except Exception:
    signal_bus = None
# ==============================
# [新增] 注意力模块定义
# ==============================
class SEAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.avg_pool(x))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = SEAttention(in_planes, ratio)
        # 修正: CBAM的空间注意力部分应该独立于SE
        class SpatialAttention(nn.Module):
            def __init__(self, kernel_size=7):
                super().__init__()
                self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                x = torch.cat([avg_out, max_out], dim=1)
                x = self.conv1(x)
                return self.sigmoid(x)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = x * self.sa(x)
        return x

class NonLocalAttention(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalAttention, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, 1)
        self.W = nn.Conv2d(self.inter_channels, self.in_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        # 修正: 卷积层默认没有 bias，除非显式声明。为保持一致性，不初始化 bias。
        # nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W(y)
        return x + W_y
    
# --- 新增的注意力模块 ---

class ECAAttention(nn.Module):
    """ 高效通道注意力 (ECA-Net) """
    def __init__(self, in_planes, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return x * self.relu(x + 3) / 6

class CoordinateAttention(nn.Module):
    """ 坐标注意力 (CoordAtt) """
    def __init__(self, in_planes, out_planes, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_planes // reduction)
        self.conv1 = nn.Conv2d(in_planes, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, out_planes, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_planes, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w)
        a_h = self.sigmoid(a_h)
        a_w = self.sigmoid(a_w)
        out = identity * a_w * a_h
        return out

class ExternalAttention(nn.Module):
    """ 外部注意力 """
    def __init__(self, in_channels, inter_channels):
        super(ExternalAttention, self).__init__()
        self.inter_channels = inter_channels
        self.conv_phi = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.conv_theta = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.k = 64 # 记忆单元的数量
        self.mu = nn.Parameter(torch.randn(1, self.inter_channels, self.k))
        self.conv_g = nn.Conv2d(self.k, in_channels, 1, bias=False)

    def forward(self, x):
        n, c, h, w = x.shape
        phi = self.conv_phi(x).view(n, self.inter_channels, -1) # N, C', H*W
        theta = self.conv_theta(x).view(n, self.inter_channels, -1).permute(0, 2, 1) # N, H*W, C'
        
        attn = torch.matmul(theta, self.mu) # N, H*W, k
        attn = F.softmax(attn, dim=-1) # Softmax over k
        
        g = self.conv_g(attn.permute(0, 2, 1).view(n, self.k, h, w))
        
        return x + g
        
    

def _res_lb(y, x, eps: float):  # 残差下限
    return x * eps + y * (1.0 - eps) if eps > 0 else y

class CoordAtt_U(nn.Module):
    def __init__(self, channels: int, alpha: float = 0.5, epsilon: float = 0.1):
        super().__init__()
        self.base = CoordinateAttention(channels, channels)
        self.alpha = float(alpha); self.epsilon = float(epsilon)
    def forward(self, x):
        y = self.base(x)
        if signal_bus is not None:
            w = signal_bus.get().get("w_mcb", None)
            if w is not None:
                gain = (1.0 + self.alpha * (w.std().clamp_min(0.0))).to(dtype=y.dtype, device=y.device)
                y = y * gain
        return _res_lb(y, x, self.epsilon)

class CBAM_GE(nn.Module):
    def __init__(self, channels: int, alpha: float = 0.5, epsilon: float = 0.1):
        super().__init__()
        self.base = CBAM(channels)
        self.alpha = float(alpha); self.epsilon = float(epsilon)
    def forward(self, x):
        y = self.base(x)
        if signal_bus is not None:
            g = signal_bus.get().get("g_strength", None)
            if g is not None:
                gain = (1.0 + self.alpha * (g.std().clamp_min(0.0))).to(dtype=y.dtype, device=y.device)
                y = y * gain
        return _res_lb(y, x, self.epsilon)

class ECA_DA(nn.Module):
    def __init__(self, channels: int, beta: float = 0.3, epsilon: float = 0.1):
        super().__init__()
        self.base = ECAAttention(channels)
        self.beta = float(beta); self.epsilon = float(epsilon)
    def forward(self, x):
        y = self.base(x)
        if signal_bus is not None:
            d = signal_bus.get().get("difficulty", None)
            if d is not None:
                gain = (1.0 + self.beta * d).to(dtype=y.dtype, device=y.device)
                y = y * gain
        return _res_lb(y, x, self.epsilon)



ATTENTION_REGISTRY = {
    "se": SEAttention,
    "cbam": CBAM,
    "nonlocal": NonLocalAttention,
    "eca": ECAAttention,
    "coordatt": CoordinateAttention,
    "external": ExternalAttention,
    "coordatt_u": CoordAtt_U,   # 不确定性偏置（N3）
    "cbam_ge": CBAM_GE,         # 梯度均衡偏置（N4）
    "eca_da": ECA_DA,           # 难度驱动+残差下限（N5）
}