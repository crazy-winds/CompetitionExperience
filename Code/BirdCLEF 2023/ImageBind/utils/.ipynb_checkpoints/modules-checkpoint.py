import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import update_dropout_rate

import sys
sys.path.append("/home/zhy/Bird/ImageBind")
from models.imagebind_model import imagebind_huge


class ConvNext(nn.Module):
    def __init__(
        self,
        num_classes=1,
        pretrained=True,
        dropout=.3,
        in_channels=1,
    ):
        super().__init__()
        self.model = timm.create_model("convnextv2_tiny", pretrained=pretrained, in_chans=in_channels, num_classes=num_classes, drop_rate=dropout, drop_path_rate=dropout)
        last_channels = self.model.head.fc.in_features
        self.model.head.fc = nn.Sequential(
            nn.LayerNorm(last_channels),
            nn.Linear(last_channels, 768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(768),
            nn.Linear(768, num_classes)
        )
        
        for module in [self.model.head.fc]:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                    
    def forward(self, x):
        return self.model(x)

    
class ImageBindAudio(nn.Module):
    AUDIO = "audio"
    def __init__(
        self,
        classes=264,
        checkpoint_path="checkpoints/imagebind_huge.pth",
    ):
        super().__init__()
        model = imagebind_huge(False)
        if checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path))
        self.preprocessors = model.modality_preprocessors[self.AUDIO]
        self.trunk = model.modality_trunks[self.AUDIO]
        self.head = model.modality_heads[self.AUDIO]
        self.postprocessors = model.modality_postprocessors[self.AUDIO]
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(.1),
            nn.Linear(512, classes)
        )
        
        # for module in [self.preprocessors, self.trunk, self.head, self.postprocessors]:
        # for module in (self.trunk, ):
        #     for m in module.parameters():
        #         m.requires_grad_(False)
                
        for module in [self.fc, ]:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
        
    def forward(self, x):
        value = self.preprocessors(x)
        trunk = value["trunk"]
        head = value["head"]
        value = self.trunk(**trunk)
        value = self.head(value, **head)
        value = self.postprocessors(value)
        return self.fc(value)
    
    def load_state_dict(self, pt_path: str, map_location: str="cpu"):
        checkpoint = torch.load(pt_path, map_location)
        
        self.preprocessors.load_state_dict(checkpoint["preprocessors"])
        self.trunk.load_state_dict(checkpoint["trunk"])
        self.head.load_state_dict(checkpoint["head"])
        self.postprocessors.load_state_dict(checkpoint["postprocessors"])
        self.fc.load_state_dict(checkpoint["fc"])
        print("<All keys matched successfully>")