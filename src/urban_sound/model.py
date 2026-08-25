"""DenseNet model adapted to ten environmental sound classes."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


class UrbanSoundDenseNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = densenet121(weights=weights)
        input_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_features, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True
