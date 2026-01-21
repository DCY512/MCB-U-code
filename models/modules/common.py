import torch.nn as nn

class Conv1x1(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)