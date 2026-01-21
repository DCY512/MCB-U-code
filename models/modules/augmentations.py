# models/modules/augmentations.py (最终完整版)

import random
from PIL import Image, ImageOps, ImageEnhance
import torch
from torchvision import transforms

# ==============================================================================
#  增强操作的具体实现 (供两个模块共同使用)
# ==============================================================================
def _apply_op(img: Image.Image, op_name: str, magnitude: float, interpolation, fill):
    if op_name == "Identity": return img
    elif op_name == "ShearX": return img.transform(img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0), interpolation, fill)
    elif op_name == "ShearY": return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0), interpolation, fill)
    elif op_name == "TranslateX": return img.transform(img.size, Image.AFFINE, (1, 0, magnitude, 0, 1, 0), interpolation, fill)
    elif op_name == "TranslateY": return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude), interpolation, fill)
    elif op_name == "Rotate": return img.rotate(magnitude, interpolation, fill)
    elif op_name == "Brightness": return ImageEnhance.Brightness(img).enhance(magnitude)
    elif op_name == "Color": return ImageEnhance.Color(img).enhance(magnitude)
    elif op_name == "Contrast": return ImageEnhance.Contrast(img).enhance(magnitude)
    elif op_name == "Sharpness": return ImageEnhance.Sharpness(img).enhance(magnitude)
    elif op_name == "Posterize": return ImageOps.posterize(img, int(magnitude))
    elif op_name == "Solarize": return ImageOps.solarize(img, int(magnitude))
    elif op_name == "AutoContrast": return ImageOps.autocontrast(img)
    elif op_name == "Equalize": return ImageOps.equalize(img)
    else: raise ValueError(f"Unknown operation {op_name}")

# --- 定义完整的操作空间 ---
AUGMENTATION_SPACE = {
    # 操作名: (强度下限, 强度上限)
    "ShearX": (-0.3, 0.3), "ShearY": (-0.3, 0.3),
    "TranslateX": (-150.0 / 331.0, 150.0 / 331.0), # 强度表示为图像尺寸的百分比
    "TranslateY": (-150.0 / 331.0, 150.0 / 331.0),
    "Rotate": (-30.0, 30.0), "Brightness": (0.05, 0.95),
    "Color": (0.05, 0.95), "Contrast": (0.05, 0.95),
    "Sharpness": (0.05, 0.95), "Posterize": (4, 8),
    "Solarize": (0, 255), "AutoContrast": (0, 0),
    "Equalize": (0, 0), "Identity": (0, 0),
}

# ==============================
# 模块一: RandAugment
# ==============================
class RandAugment(torch.nn.Module):
    def __init__(self, n: int, m: int, interpolation=Image.BICUBIC, fill=None):
        super().__init__()
        self.n = n # 每次随机选择 n 个操作
        self.m = m # 所有操作的强度 (0-30)
        self.interpolation = interpolation
        self.fill = fill
        self.op_list = list(AUGMENTATION_SPACE.keys())

    def forward(self, img: Image.Image) -> Image.Image:
        ops = random.choices(self.op_list, k=self.n)
        for op_name in ops:
            min_val, max_val = AUGMENTATION_SPACE[op_name]
            # 根据强度 M (0-30)，计算出当前操作的具体参数值
            magnitude = (float(self.m) / 30.0) * (max_val - min_val) + min_val
            
            # 特殊处理 Translate, 因为它的强度是像素值
            if op_name.startswith("Translate"):
                magnitude *= img.size[0] if op_name.endswith("X") else img.size[1]

            img = _apply_op(img, op_name, magnitude, self.interpolation, self.fill)
        return img

# ==============================
# 模块二: TrivialAugmentWide
# ==============================
class TrivialAugmentWide(torch.nn.Module):
    def __init__(self, num_magnitude_bins=31, interpolation=Image.BICUBIC, fill=None):
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.op_list = list(AUGMENTATION_SPACE.keys())

    def forward(self, img: Image.Image) -> Image.Image:
        op_name = random.choice(self.op_list)
        min_val, max_val = AUGMENTATION_SPACE[op_name]
        
        magnitude_values = torch.linspace(min_val, max_val, self.num_magnitude_bins).tolist()
        magnitude = random.choice(magnitude_values)

        if op_name.startswith("Translate"):
            magnitude *= img.size[0] if op_name.endswith("X") else img.size[1]
            
        return _apply_op(img, op_name, magnitude, self.interpolation, self.fill)