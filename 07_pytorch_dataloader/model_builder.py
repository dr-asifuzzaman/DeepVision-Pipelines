# model_builder.py
import torch
import torch.nn as nn
from torchvision import models


def build_mobilenet_v2(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
