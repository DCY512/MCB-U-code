import numpy.random as random

import torch
import torch.nn as nn
import torch.nn.functional as F
# [ADDED] 可选导入 MinkowskiEngine；2D 训练不需要它
try:
    from MinkowskiEngine import SparseTensor  # type: ignore
    ME_AVAILABLE = True
except Exception:
    class SparseTensor:  # 简单占位，确保类型检查不报错；真正用到时再报友好错误
        pass
    ME_AVAILABLE = False


# ======================
# [ADDED] 小工具：在未安装 ME 时抛出更友好的错误
# ======================
def _require_me():
    if not ME_AVAILABLE:
        raise ImportError(
            "MinkowskiEngine 未安装。该模块仅用于稀疏/3D 场景；"
            "请避免在 2D 任务中实例化 Minkowski* 层，或按官方文档安装 ME。"
        )

class MinkowskiGRN(nn.Module):
    """ GRN layer for sparse tensors. """
    def __init__(self, dim):  
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        _require_me()  # [ADDED]
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key

        Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return SparseTensor(
                self.gamma * (x.F * Nx) + self.beta + x.F,
                coordinate_map_key=in_key,
                coordinate_manager=cm)

class MinkowskiDropPath(nn.Module):
    """ Drop Path for sparse tensors. """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(MinkowskiDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        _require_me()  # [ADDED]
        if self.drop_prob == 0. or not self.training:
            return x
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key
        keep_prob = 1 - self.drop_prob

        # [CHANGED] 用 torch.rand 直接在设备上采样掩码（替代 numpy.random）
        mask_list = []
        for coords in x.decomposed_coordinates:
            # 每个子块一次伯努利采样（保留或丢弃）
            keep = (torch.rand((), device=x.device) > self.drop_prob)
            m = torch.ones(len(coords), 1, device=x.device) if keep else torch.zeros(len(coords), 1, device=x.device)
            mask_list.append(m)
        mask = torch.cat(mask_list, dim=0)
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)

        return SparseTensor(
                x.F * mask,
                coordinate_map_key=in_key,
                coordinate_manager=cm)

class MinkowskiLayerNorm(nn.Module):
    """ Channel-wise layer normalization for sparse tensors. """
    def __init__(self, normalized_shape, eps=1e-6):
        super(MinkowskiLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)
    def forward(self, input):
        _require_me()  # [ADDED]
        output = self.ln(input.F)
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager)
            
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
