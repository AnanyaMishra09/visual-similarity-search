"""ResNet-18 embedding backbone, built from scratch (no torchvision.models,
no pretrained weights, just nn.Conv2d / nn.BatchNorm2d / nn.Linear).

Architecture is fixed per the assignment: 4 residual stages, global average
pool, final FC replaced by a single nn.Linear -> 128-d, L2-normalised output.

One deliberate deviation, which the assignment leaves open (backbone =
"ResNet-18 ... or a custom CNN", stages/GAP/head are the only hard
constraints): the stem. The textbook ImageNet ResNet-18 stem is a 7x7
stride-2 conv + a 3x3 stride-2 maxpool, which crushes a 32x32 CIFAR image
down to 8x8 *before* a single residual block runs. That throws away most
of the spatial detail a 32x32 image has to begin with. I used the standard
CIFAR stem instead (3x3 stride-1 conv, no maxpool), which is the
well-established adaptation used by every CIFAR-ResNet paper (He et al.
2016 use it for their CIFAR experiments directly). It still is a 4-stage
ResNet-18 with the same block structure and channel widths. The only
change is how much spatial information survives before the first block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out, inplace=True)


class ResNet18Embedding(nn.Module):
    """4-stage ResNet-18 -> GAP -> Linear(512, embed_dim) -> L2 normalize."""

    def __init__(self, embed_dim: int = 128, in_channels: int = 3):
        super().__init__()
        self.in_ch = 64
        # CIFAR stem: 3x3 stride-1, no maxpool (see module docstring).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(64, 2, stride=1)
        self.stage2 = self._make_stage(128, 2, stride=2)
        self.stage3 = self._make_stage(256, 2, stride=2)
        self.stage4 = self._make_stage(512, 2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embed_dim)

        self._init_weights()

    def _make_stage(self, out_ch: int, n_blocks: int, stride: int):
        layers = [BasicBlock(self.in_ch, out_ch, stride=stride)]
        self.in_ch = out_ch
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(self.in_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def build_model(embed_dim: int = 128) -> ResNet18Embedding:
    return ResNet18Embedding(embed_dim=embed_dim)
