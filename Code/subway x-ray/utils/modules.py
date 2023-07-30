import torch
import torch.nn as nn

import mmengine
from mmdet.registry import MODELS as MMDET_MODELS

import random


class Model(nn.Module):
    def __init__(
        self,
        cfg_path=None,
    ):
        """ 初始化模型
        
        Args:
            cfg_path (str): 模型配置文件
        """
        super().__init__()
        cfg = mmengine.Config.fromfile(cfg_path)
        self.model = MMDET_MODELS.build(cfg.model)
        
        if cfg.load_from:
            pth = torch.load(cfg.load_from,"cpu")["state_dict"]
            for i, (name, param) in enumerate(self.model.named_parameters(), 1):
                if name not in pth:
                    print("Missing key(s) in state_dict :{}".format(name))
                    if param.dim() > 1:
                        nn.init.xavier_normal_(param)
                    continue
                
                try:
                    param.detach().copy_(pth[name])
                except RuntimeError:
                    param.detach().copy_(pth[name].resize_(param.shape))
                
            print("All Keys Matching")
        
    def forward(self, data, mode="loss"):
        out = self.model(**data, mode=mode)
        if mode == "loss":
            # return sum(out["loss_cls"]), sum(out["loss_bbox"])
            # return (
            #     sum(out[key] for key in out.keys() if "cls" in key and "loss" in key),
            #     sum(out[key] for key in out.keys() if "cls" not in key and "loss" in key),
            # )
        
            return (
                out["loss_cls"][0] * 1 + out["loss_cls"][1] * 1. + out["loss_cls"][2] * 1.,
                out["loss_bbox"][0] * 1 + out["loss_bbox"][1] * 1. + out["loss_bbox"][2] * 1.,
            )
        
        return out
