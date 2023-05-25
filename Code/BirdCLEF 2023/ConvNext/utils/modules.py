import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["ConvNext"]


class ConvNext(nn.Module):
    def __init__(
        self,
        num_classes=1,
        pretrained=True,
        dropout=.3,
    ):
        super().__init__()
        # self.model = timm.create_model("convnextv2_base", pretrained=pretrained, in_chans=1, num_classes=num_classes, drop_rate=dropout, drop_path_rate=dropout)
        self.model = timm.create_model("convnextv2_tiny", pretrained=pretrained, in_chans=1, num_classes=num_classes, drop_rate=dropout, drop_path_rate=dropout)
        last_channels = self.model.head.fc.in_features
        # self.model.head.global_pool = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((4, 4)),
        #     nn.Conv2d(last_channels, last_channels, 4),
        #     nn.LayerNorm(last_channels),
        #     nn.ReLU(),
        # )
        self.model.head.fc = nn.Sequential(
            nn.Linear(last_channels, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, num_classes)
        )
        
        for module in [self.model.head.fc, self.model.head.global_pool]:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                    
    def forward(self, x):
        return self.model(x)

    
class SoftPool(nn.Module):
    def __init__(self, size=(4, 12)):
        super().__init__()
        self.p = nn.Parameter((1, 1, *size))
        
    def forward(self, x):
        weights = F.softmax(self.p * x, axis=(-1, -2))
        return F.sum(x * weights, axis=-1, keepdims=True)