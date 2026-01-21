# models/modules/distillation/predictors.py

from typing import Dict, List
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """
    一个通用的特征提取器，能够通过 PyTorch Hooks 抓取任何 nn.Module 的中间层特征。
    """
    def __init__(self, model: nn.Module, target_layer_names: List[str]):
        """
        Args:
            model: 要从中提取特征的模型 (例如 backbone)。
            target_layer_names: 一个包含层名称的列表, e.g., ['stages.0', 'stages.1']。
                                 您可以通过 `model.named_modules()` 查看所有可用的层名称。
        """
        super().__init__()
        self.model = model
        self.target_layers = target_layer_names
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks = []

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                # 注册一个 forward hook
                hook = module.register_forward_hook(self._save_feature_hook(name))
                self._hooks.append(hook)

    def _save_feature_hook(self, name: str):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        执行模型的前向传播，并返回一个包含所有目标层特征的字典。
        """
        self.features.clear()  # 清除上一次的特征
        # 只需要执行模型的前向传播，hook 会自动填充 self.features
        _ = self.model(x) 
        return self.features

    def remove_hooks(self):
        """
        在提取完成后，清理所有注册的 hooks，避免内存泄漏。
        """
        for hook in self._hooks:
            hook.remove()