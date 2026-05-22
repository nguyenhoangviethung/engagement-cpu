import torch
from torch import nn
from torchvision import models


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.view(-1)


def build_cnn_model(model_name: str = "mobilenet_v3_small", pretrained: bool = False, freeze_backbone: bool = False) -> nn.Module:
    model_name = model_name.strip().lower()

    if model_name == "tinycnn":
        return TinyCNN()

    if model_name == "mobilenet_v3_small":
        weights = None
        if pretrained:
            try:
                from torchvision.models import MobileNet_V3_Small_Weights

                weights = MobileNet_V3_Small_Weights.DEFAULT
            except Exception:
                weights = None
        model = models.mobilenet_v3_small(weights=weights)
        if freeze_backbone:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return model

    if model_name == "efficientnet_b0":
        weights = None
        if pretrained:
            try:
                from torchvision.models import EfficientNet_B0_Weights

                weights = EfficientNet_B0_Weights.DEFAULT
            except Exception:
                weights = None
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return model

    raise ValueError(f"Unsupported model name: {model_name}")

