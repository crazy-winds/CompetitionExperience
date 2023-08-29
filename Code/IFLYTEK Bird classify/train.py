#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math
import tqdm
import numpy as np
import pandas as pd
from timm.optim.adan import Adan
from timm.utils.model_ema import ModelEmaV2
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss.asymmetric_loss import AsymmetricLossMultiLabel

from accelerate import Accelerator
import albumentations as A
import audiomentations as Audio
from albumentations.pytorch import ToTensorV2

from utils import (
    util,
    modules,
    custom_dataset,
)

import gc
import sklearn.metrics


torch.set_float32_matmul_precision('high')


# In[ ]:


accelerator = Accelerator()
device = accelerator.device


# # Parameters

# In[ ]:


MODEL_NAME = "ConvNextT"

IMAGE_SIZE = 420
LR = 2e-5
LR_MIN = 1e-5
EPOCHS = 30
VALID_EPOCH = 3
BATCH_SIZE = 64
NUM_WORKER = 6
WEIGHT_DECAY = 3e-4
TRAIN_DATA_PATH = "../my-inference-script-and-model/datasets/train/"
VALID_DATA_PATH = "../my-inference-script-and-model/datasets/val/"

cur_epoch = 1
# swa_start = EPOCHS // 5 * 3
swa_start = 1
swa_freq = 5

train_transform = A.Compose(
    [
        A.OneOf(
            [
                A.SmallestMaxSize(max_size=640),
                # A.SmallestMaxSize(max_size=720),
                A.SmallestMaxSize(max_size=512),
                A.SmallestMaxSize(max_size=480),
                A.SmallestMaxSize(max_size=IMAGE_SIZE),
            ],
            p=1.
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(30, p=.5),
        # A.ShiftScaleRotate(rotate_limit=60, scale_limit=.1, p=.6),
        A.PixelDropout(drop_value=255),
        A.GaussNoise(p=.5),
        # A.RandomBrightnessContrast(.4, .4, p=.4),
        A.OneOf([
            A.CLAHE(),
            A.IAASharpen(),
            A.IAAEmboss(),
        ], p=0.1),
        A.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        # A.CoarseDropout(max_holes=40, max_height=32, max_width=32, fill_value=255, p=0.8),
        A.Normalize(),
        ToTensorV2()
    ]
)
test_transform = A.Compose([
    A.SmallestMaxSize(max_size=IMAGE_SIZE),
    A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# # Dataset

def collate_fn(batch):
    x = torch.stack([b[0] for b in batch])
    y = [b[1] for b in batch]
    
    return x, y



train_dataloader = torch.utils.data.DataLoader(
    custom_dataset.CustomDatset(
        TRAIN_DATA_PATH,
        transform=train_transform,
    ),
    # collate_fn=collate_fn,
    batch_size=BATCH_SIZE,
    pin_memory=True,
    shuffle=True,
    num_workers=NUM_WORKER,
)

valid_dataloader = torch.utils.data.DataLoader(
    custom_dataset.CustomDatset(
        VALID_DATA_PATH,
        transform=test_transform,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKER,
)


# # Model

# In[ ]:
model = modules.ConvNext(25, pretrained=True, dropout=0.1)
# model.load_state_dict(torch.load("checkpoint/DINOv2_sota.pt", "cpu")["model"])

# # Misc

# In[ ]:
class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits

    
class Criterion(nn.CrossEntropyLoss):
    def __init__(self, sample_num=3, **kwargs):
        super().__init__(**kwargs)
        self.reduction = "none"
        self.sample_num = sample_num
        
    def forward(self, x, y):
        # if isinstance(y, list):
        #     m = torch.zeros_like(x, device=device)
        #     m.scatter_(0, y[1]) = y[3]
        #     m.scatter_(0, y[2]) = 1 - y[3]
        #     x = F.log_softmax(x, dim=-1)
        #     loss = F.nll_loss(x, m)
        # else:
        loss = super().forward(x, y)
            
        ohem_loss = loss.sort(-1, True).values[:self.sample_num]
        return loss.mean() + ohem_loss.mean() * .7

# label_smoothing=.1
criterion = nn.CrossEntropyLoss()
# criterion = Criterion(5)
optimizer = Adan(
    util.get_param_groups(model, ("norm",)),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
lr_scheduler = CosineLRScheduler(
    optimizer,
    EPOCHS,
    warmup_lr_init=LR_MIN,
    warmup_t=EPOCHS // 6,
)


# In[ ]:


model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
    model, 
    optimizer,
    train_dataloader,
    valid_dataloader,
    lr_scheduler
)


# # Train

# In[ ]:


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    dataloader = tqdm.tqdm(dataloader, disable=not accelerator.is_main_process)
    
    losses = 0
    for i, (x, y) in enumerate(dataloader, 1):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        losses += loss.item()
        
        accelerator.backward(loss)
        
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        optimizer.step()
    
    return losses / len(dataloader)


# In[ ]:


@torch.no_grad()
def valid_one_epoch(model, dataloader):
    model.eval()
    dataloader = tqdm.tqdm(dataloader, disable=not accelerator.is_main_process)
    
    acc = 0
    for x, y in dataloader:
        out = model(x)
        acc += ((out.argmax(-1) == y).sum() / x.size(0)).item()
    
    acc /= len(dataloader)
    return acc * 100


# In[ ]:


min_loss = float("inf")
max_score = 0
for epoch in range(cur_epoch, EPOCHS + 1):
    # # # # #
    # Train #
    # # # # #
    loss = train_one_epoch(model, train_dataloader, optimizer, criterion)
    accelerator.wait_for_everyone()
    lr_scheduler.step(epoch)
    accelerator.print(f"Epoch {epoch} - Loss: {loss :.4f}\tlr: {optimizer.param_groups[0]['lr']*1e4 :.4f}")
    if min_loss > loss:
        min_loss = loss
    # if epoch == int(EPOCHS // 6):
    #     util.update_dropout_rate(model, .0)
    if accelerator.is_local_main_process:
        torch.save(
            {
                "model": accelerator.unwrap_model(model).state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            },
            f"checkpoint/{MODEL_NAME}_all.pt"
        )

        torch.save(
            {
                "model": accelerator.unwrap_model(model).state_dict(),
                # "ema": ema_model.module.state_dict()
            },
            f"checkpoint/{MODEL_NAME}_model.pt"
        )
        
    # # # # #
    # Valid #
    # # # # #
    if epoch % VALID_EPOCH == 0:
        s = valid_one_epoch(model, valid_dataloader)
        if s > max_score and accelerator.is_local_main_process:
            torch.save(
                {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    # "ema": ema_model.module.state_dict()
                },
                f"checkpoint/{MODEL_NAME}_best.pt"
            )
        max_score = max(s, max_score)
        accelerator.print(f"Valid Epoch {epoch} - Score: {s :.2f}%\tMax Score: {max_score :.2f}%")
        
    # # # # #
    # # SWA #
    # # # # #
    if swa_start == epoch:
        swa_model = torch.optim.swa_utils.AveragedModel(accelerator.unwrap_model(model))
    if swa_start <= epoch and epoch % swa_freq == 0:
        swa_model.update_parameters(model)
        torch.save(
            {
                "model": swa_model.module.state_dict()
            },
            f"checkpoint/{MODEL_NAME}_swa.pt"
        )

