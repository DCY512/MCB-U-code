from __future__ import annotations
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
# --- 从新的模块文件中导入所有“零件” ---
from .modules.common import Conv1x1
from .modules.fusions import GatedFuse, XAttnFuse, AHCRFuse
from .modules.necks import FPN, FPN_PAN

# ------------------------------
# Dual Wrapper
# ------------------------------
class ConvNeXtV2Dual(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        ahcr_mode: str = 'intra_level',
        return_intermediate: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3),
        fuse_mode: str = "concat",
        fuse_levels: Tuple[str, ...] = ("C3", "C4", "C5"),
        head_type: str = "c5",
        xattn_heads: int = 4,
        xattn_reduction: int = 4,
        fpn_out_channels: int = 256,
        # 新增的参数，由 main_finetune.py 传递
        attention_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.return_intermediate = bool(return_intermediate)
        self.out_indices = tuple(out_indices)
        self.fuse_mode = str(fuse_mode)
        self.fuse_levels = tuple(fuse_levels)
        self.head_type = str(head_type) # head_type 现在是 'fpn_pan' 等基础类型
        self.attention_config = attention_config
        self.xattn_heads = int(xattn_heads)
        self.xattn_reduction = int(xattn_reduction)

        if not hasattr(backbone, "dims"):
            raise AttributeError("backbone 模型缺少属性 'dims'，请检查 models/convnextv2.py 实现。")
        dims: List[int] = list(backbone.dims)

        self.stage_to_name = {1: "C3", 2: "C4", 3: "C5"}
        self.name_to_dim = {self.stage_to_name[i]: dims[i] for i in self.out_indices}

        self.xattn_fuses = nn.ModuleDict()
        self.fuse_convs = nn.ModuleDict()
        self.gated_fuses = nn.ModuleDict()

        if self.fuse_mode in ("concat", "gated"):
            for name in self.fuse_levels:
                if name not in self.name_to_dim: continue
                C = self.name_to_dim[name]
                self.fuse_convs[name] = Conv1x1(in_ch=2 * C, out_ch=C, act=True)
                if self.fuse_mode == "gated":
                    self.gated_fuses[name] = GatedFuse(in_ch=2 * C, out_ch=C)
        elif self.fuse_mode == "xattn":
            for name in self.fuse_levels:
                if name not in self.name_to_dim: continue
                C = self.name_to_dim[name]
                self.xattn_fuses[name] = XAttnFuse(
                    channels=C, num_heads=self.xattn_heads, reduction=self.xattn_reduction
                )
        elif self.fuse_mode == "ahcr":
            c3, c4, c5 = self.name_to_dim["C3"], self.name_to_dim["C4"], self.name_to_dim["C5"]
            self.ahcr_fuser = AHCRFuse(c3, c4, c5, mode=ahcr_mode)

        self.fpn = None
        self.fpn_pan = None
        ht = self.head_type.lower()
        if ht == "fpn_fuse":
            c3, c4, c5 = self.name_to_dim["C3"], self.name_to_dim["C4"], self.name_to_dim["C5"]
            self.fpn = FPN(c3, c4, c5, out_channels=fpn_out_channels)
        elif ht == "fpn_pan":
            c3, c4, c5 = self.name_to_dim["C3"], self.name_to_dim["C4"], self.name_to_dim["C5"]
            self.fpn_pan = FPN_PAN(c3, c4, c5, out_channels=fpn_out_channels, attention_config=self.attention_config)

        if ht == "c5":
            feat_ch = self.name_to_dim["C5"]
        elif ht == "fpn":
            feat_ch = sum(self.name_to_dim.get(n, 0) for n in self.fuse_levels)
        elif ht in ("fpn_fuse", "fpn_pan"):
            feat_ch = fpn_out_channels
        else:
            raise ValueError(f"head_type 必须是 'c5', 'fpn', 'fpn_fuse', 或 'fpn_pan', 收到: {ht}")

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feat_ch, self.num_classes) if self.num_classes > 0 else nn.Identity()

    # 提取单视角多层特征
    def _extract_single_view(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            feats: Dict[str, torch.Tensor] = {}
            
            # --- 【核心修改】智能判断 V1/V2 架构 ---
            # 假设 V1 模型有 forward_features 方法，而 V2 模型没有（或实现不同）
            # 一个更可靠的判断是检查模块名称
            is_v1_style = 'ConvNeXt' == self.backbone.__class__.__name__

            if is_v1_style:
                # V1 模式: forward_features 返回一个包含 4 个特征图的列表
                feature_list = self.backbone.forward_features(x)
                # V1 的 stage 0,1,2,3 分别对应 C2,C3,C4,C5
                # 我们的 out_indices 1,2,3 对应列表索引 1,2,3
                for i in self.out_indices: # 1, 2, 3
                    if i < len(feature_list):
                        feats[self.stage_to_name[i]] = feature_list[i]
            else:
                # V2 模式: 保持原来的逻辑
                # (注意：V2 的 downsample 和 stage 也是分开的，但逻辑略有不同)
                x = self.backbone.downsample_layers[0](x)
                x = self.backbone.stages[0](x)
                # 这里的 self.out_indices (1,2,3) 对应 V2 的 stage 1,2,3
                for i in range(1, 4):
                    x = self.backbone.downsample_layers[i](x)
                    x = self.backbone.stages[i](x)
                    if i in self.out_indices:
                        feats[self.stage_to_name[i]] = x
            
            if not feats:
                raise RuntimeError("错误: _extract_single_view 未能从 backbone 提取任何特征。请检查 V1/V2 兼容性逻辑。")
                
            return feats
    # 融合两路特征
    def _fuse_pair(self, xa: torch.Tensor, xb: torch.Tensor, name: str) -> torch.Tensor:
        # 注意：这个方法现在不再处理 ahcr 模式
        if name not in self.fuse_levels: return xa
        if self.fuse_mode == "add": return xa + xb
        elif self.fuse_mode == "concat": return self.fuse_convs[name](torch.cat([xa, xb], dim=1))
        elif self.fuse_mode == "gated": return self.gated_fuses[name](xa, xb)
        elif self.fuse_mode == "xattn":
            if name in self.xattn_fuses: return self.xattn_fuses[name](xa, xb)
            else: return xa
        # 注意：这里不再有 ahcr 的 elif 分支
        else: raise ValueError(f"未知或不应在此处理的 fuse_mode: {self.fuse_mode}")

    def _name_to_stage(self, name: str) -> int:
        for k, v in self.stage_to_name.items():
            if v == name:
                return k
        raise KeyError(name)

    def forward(self, xa: torch.Tensor, xb: Optional[torch.Tensor] = None):
        feats_a = self._extract_single_view(xa)
        feats_b = self._extract_single_view(xb) if xb is not None else feats_a

        fused: Dict[str, torch.Tensor] = {}
        if self.fuse_mode == "ahcr":
            fused = self.ahcr_fuser(feats_a, feats_b)
        else:
            for name in self.name_to_dim:
                if name in feats_a and name in feats_b:
                    fused[name] = self._fuse_pair(feats_a[name], feats_b[name], name)

        ht = self.head_type.lower()
        x = None
        if ht == "c5":
            x = self.global_pool(fused["C5"]).flatten(1)
        elif ht == "fpn":
            vecs = [self.global_pool(fused[n]).flatten(1) for n in self.fuse_levels if n in fused]
            x = torch.cat(vecs, dim=1) if len(vecs) > 1 else vecs[0]
        elif self.fpn is not None:  # 处理 fpn_fuse
            p3, _, _ = self.fpn(fused["C3"], fused["C4"], fused["C5"])
            x = self.global_pool(p3).flatten(1)
        elif self.fpn_pan is not None:  # 处理 fpn_pan 及其所有注意力变体
            n3, _, _ = self.fpn_pan(fused["C3"], fused["C4"], fused["C5"])
            x = self.global_pool(n3).flatten(1)
        else:
            raise ValueError(f"未知 head_type: {self.head_type}")

        logits = self.classifier(x)
        if not self.return_intermediate:
            return logits
        return {"logits": logits, "feats": {"A": feats_a, "B": feats_b, "fused": fused}}